import numpy as np
import pandas as pd
import pytest

from disputes_ml.config import DAY, HOUR, REF_EPOCH
from disputes_ml.features_ref import (
    asof_lookup_pd,
    card_uid,
    compute_card_features_pd,
    incremental_update_pd,
    prepare_pd,
)
from disputes_ml.synthetic import generate


def tx(rows):
    """rows: (card, epoch_offset_seconds, amount, email, product)"""
    return pd.DataFrame(
        [{"card_uid": c, "event_epoch": REF_EPOCH + t, "TransactionAmt": a,
          "P_emaildomain": e, "ProductCD": p} for c, t, a, e, p in rows]
    )


def test_card_uid_handles_missing_parts():
    assert card_uid(1234, 315.0, 14.0, 20 * DAY + 5) == "1234_315_6"
    assert card_uid(1234, np.nan, np.nan, 5) == "1234_NA_NA"


def test_prepare_sets_epoch_from_reference():
    raw = generate(n_cards=5, days=10, seed=1)
    p = prepare_pd(raw)
    assert (p["event_epoch"] == REF_EPOCH + raw["TransactionDT"]).all()
    # day - D1 is the account start day, constant per synthetic card
    assert p.groupby("card1")["card_uid"].nunique().max() <= 1


def test_windows_are_half_open_and_include_current():
    f = compute_card_features_pd(tx([
        ("a", 0, 10.0, "x.com", "W"),
        ("a", HOUR, 20.0, "y.com", "W"),       # exactly 1h later: first txn falls out of 1h
        ("a", HOUR + 1, 30.0, "x.com", "C"),
    ]))
    assert f["txn_cnt_1h"].tolist() == [1, 1, 2]
    assert f["txn_cnt_24h"].tolist() == [1, 2, 3]
    assert f["amt_sum_24h"].tolist() == [10.0, 30.0, 60.0]
    assert np.isnan(f["amt_std_7d"].iloc[0])
    assert f["n_email_7d"].tolist() == [1, 2, 2]
    assert f["n_product_7d"].tolist() == [1, 1, 2]


def test_same_second_transactions_collapse_to_one_row():
    f = compute_card_features_pd(tx([("a", 0, 10.0, "x", "W"), ("a", 0, 5.0, "x", "W")]))
    assert len(f) == 1
    assert f["txn_cnt_1h"].iloc[0] == 2
    assert f["amt_sum_24h"].iloc[0] == 15.0


def test_point_in_time_lookup_never_sees_current_transaction():
    t = tx([("a", 0, 10.0, "x", "W"), ("a", 100, 99.0, "x", "W"), ("b", 50, 1.0, "z", "W")])
    feats = compute_card_features_pd(t)
    spine = t.assign(lookup_epoch=t["event_epoch"] - 1)
    out = asof_lookup_pd(spine, feats, "lookup_epoch")
    # first txn of each card has no history
    assert np.isnan(out.loc[0, "txn_cnt_24h"]) and np.isnan(out.loc[2, "txn_cnt_24h"])
    # second txn on card a sees only the first one
    assert out.loc[1, "txn_cnt_24h"] == 1
    assert out.loc[1, "amt_max_7d"] == 10.0
    assert out.loc[1, "last_txn_epoch"] == REF_EPOCH


def test_old_events_drop_out_of_7d_window():
    f = compute_card_features_pd(tx([("a", 0, 10.0, "x", "W"), ("a", 7 * DAY, 1.0, "y", "W")]))
    assert f["txn_cnt_7d"].tolist() == [1, 1]
    assert f["n_email_7d"].tolist() == [1, 1]


@pytest.mark.parametrize("shuffle", [False, True])
def test_incremental_stream_updates_equal_full_recompute(shuffle):
    raw = prepare_pd(generate(n_cards=60, days=30, seed=3, drift_from_day=None))
    order = raw.sample(frac=1, random_state=0) if shuffle else raw  # shuffle = out-of-order
    full = compute_card_features_pd(raw)

    first, rest = order.iloc[:200], order.iloc[200:]
    history = first.copy()
    table = compute_card_features_pd(history)
    for start in range(0, len(rest), 137):
        batch = rest.iloc[start:start + 137]
        history = pd.concat([history, batch], ignore_index=True)
        table = incremental_update_pd(table, history, batch)

    pd.testing.assert_frame_equal(
        table.reset_index(drop=True), full.reset_index(drop=True), check_dtype=False
    )

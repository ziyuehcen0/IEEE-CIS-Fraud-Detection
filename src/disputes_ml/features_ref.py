"""pandas reference implementation of the feature definitions.

The PySpark code in `prep.py`, `features.py` and `streaming.py` must produce exactly
what this module produces. It exists for three reasons:
  1. unit tests that pin down the point-in-time semantics without a cluster;
  2. the parity notebook, which compares Spark output with this on a sample;
  3. a readable spec of every feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAY, MAX_LOOKBACK, REF_EPOCH, WINDOWS
from .schema import CARD_FEATURES, FEATURE_TS, KEY


def card_uid(card1, addr1, d1, transaction_dt) -> str:
    """Card identifier: card1 + billing zip + account start day (day - D1).

    IEEE-CIS has no card ID; this is the usual reconstruction. Missing parts become "NA".
    """
    day = int(np.floor(transaction_dt / DAY))
    a = "NA" if pd.isna(addr1) else str(int(addr1))
    s = "NA" if pd.isna(d1) else str(int(day - d1))
    return f"{int(card1)}_{a}_{s}"


def prepare_pd(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out[KEY] = [
        card_uid(c, a, d, t)
        for c, a, d, t in zip(out["card1"], out["addr1"], out["D1"], out["TransactionDT"])
    ]
    out["event_epoch"] = (REF_EPOCH + out["TransactionDT"]).astype("int64")
    return out


def _window_stats(ts, amt, email, prod, t):
    hi = np.searchsorted(ts, t, side="right")
    row = {}
    lo = {k: np.searchsorted(ts, t - w, side="right") for k, w in WINDOWS.items()}
    row["txn_cnt_1h"] = hi - lo["1h"]
    row["txn_cnt_24h"] = hi - lo["24h"]
    row["txn_cnt_7d"] = hi - lo["7d"]
    a24 = amt[lo["24h"]:hi]
    a7 = amt[lo["7d"]:hi]
    row["amt_sum_24h"] = float(a24.sum())
    row["amt_mean_7d"] = float(a7.mean())
    row["amt_std_7d"] = float(a7.std(ddof=1)) if len(a7) > 1 else np.nan
    row["amt_max_7d"] = float(a7.max())
    row["n_email_7d"] = int(pd.Series(email[lo["7d"]:hi]).dropna().nunique())
    row["n_product_7d"] = int(pd.Series(prod[lo["7d"]:hi]).dropna().nunique())
    row["last_txn_epoch"] = int(t)
    return row


def compute_card_features_pd(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (card_uid, feature_epoch): card state *after* all txns at that second."""
    cols = [KEY, "event_epoch", "TransactionAmt", "P_emaildomain", "ProductCD"]
    tx = tx[cols].sort_values([KEY, "event_epoch"], kind="mergesort")
    rows = []
    for uid, g in tx.groupby(KEY, sort=False):
        ts = g["event_epoch"].to_numpy(np.int64)
        amt = g["TransactionAmt"].to_numpy(float)
        email = g["P_emaildomain"].to_numpy(object)
        prod = g["ProductCD"].to_numpy(object)
        for t in np.unique(ts):
            r = _window_stats(ts, amt, email, prod, t)
            r[KEY] = uid
            r["feature_epoch"] = int(t)
            rows.append(r)
    out = pd.DataFrame(rows, columns=[KEY, "feature_epoch", *CARD_FEATURES])
    return out.sort_values([KEY, "feature_epoch"]).reset_index(drop=True)


def asof_lookup_pd(spine: pd.DataFrame, feats: pd.DataFrame, lookup_epoch_col: str) -> pd.DataFrame:
    """Point-in-time join: latest feature row with feature_epoch <= lookup time, per card.

    Same semantics as Databricks FeatureLookup(timestamp_lookup_key=...). Training passes
    lookup = event_epoch - 1 so a transaction never sees itself.
    """
    left = spine.reset_index().rename(columns={"index": "_row"})
    left = left.sort_values(lookup_epoch_col)
    right = feats.sort_values("feature_epoch")
    merged = pd.merge_asof(
        left, right, left_on=lookup_epoch_col, right_on="feature_epoch",
        by=KEY, direction="backward", allow_exact_matches=True,
    )
    return merged.sort_values("_row").drop(columns=["_row", "feature_epoch"]).reset_index(drop=True)


def incremental_update_pd(
    feature_table: pd.DataFrame, history: pd.DataFrame, batch: pd.DataFrame
) -> pd.DataFrame:
    """Mirror of the streaming foreachBatch logic in `streaming.py`.

    `history` already contains `batch` (it was appended first). For the cards in the batch,
    recompute every feature row from the batch's earliest timestamp onwards, reading back
    MAX_LOOKBACK seconds of history, then upsert. Recomputing from the earliest timestamp
    (not just the new rows) keeps the table correct when events arrive out of order.
    """
    cards = set(batch[KEY])
    min_epoch = int(batch["event_epoch"].min())
    hist = history[history[KEY].isin(cards) & (history["event_epoch"] > min_epoch - MAX_LOOKBACK)]
    new = compute_card_features_pd(hist)
    new = new[new["feature_epoch"] >= min_epoch]
    keys = [KEY, "feature_epoch"]
    kept = feature_table.merge(new[keys], on=keys, how="left", indicator=True)
    kept = kept[kept["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([kept, new], ignore_index=True)
    return out.sort_values(keys).reset_index(drop=True)


__all__ = [
    "card_uid", "prepare_pd", "compute_card_features_pd", "asof_lookup_pd",
    "incremental_update_pd", "FEATURE_TS",
]

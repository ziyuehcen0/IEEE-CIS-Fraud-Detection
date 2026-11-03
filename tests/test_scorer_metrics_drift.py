import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from disputes_ml import drift
from disputes_ml.metrics import evaluate, threshold_for_capacity
from disputes_ml.schema import CARD_FEATURES, CATEGORICAL, NUMERIC_REQUEST
from disputes_ml.scorer import MISSING_CODE, OTHER_CODE, RiskScorer, add_on_demand


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_REQUEST + CARD_FEATURES})
    df["TransactionAmt"] = rng.uniform(1, 500, n).round(2)
    df["event_epoch"] = 1_600_000_000 + np.arange(n)
    df["last_txn_epoch"] = df["event_epoch"] - 60
    for c in CATEGORICAL:
        df[c] = rng.choice(["a", "b", None], n)
    return df


def test_on_demand_features():
    df = pd.DataFrame({"TransactionAmt": [10.25, 3.0], "event_epoch": [1000, 1000],
                       "last_txn_epoch": [940, np.nan], "amt_mean_7d": [5.0, 0.0]})
    out = add_on_demand(df)
    assert out["secs_since_last_txn"].iloc[0] == 60 and np.isnan(out["secs_since_last_txn"].iloc[1])
    assert out["amt_cents"].iloc[0] == pytest.approx(0.25)
    assert out["amt_vs_mean_7d"].iloc[0] == pytest.approx(2.05)
    assert np.isnan(out["amt_vs_mean_7d"].iloc[1])  # no divide-by-zero


def test_unseen_and_missing_categories_get_reserved_codes():
    df = frame()
    s = RiskScorer(HistGradientBoostingClassifier(max_iter=5)).fit(df, np.arange(len(df)) % 2)
    probe = df.head(2).copy()
    probe.loc[probe.index[0], "ProductCD"] = "never-seen"
    probe.loc[probe.index[1], "ProductCD"] = None
    x = s.transform(probe)
    assert x["ProductCD"].tolist() == [OTHER_CODE, MISSING_CODE]
    assert list(x.columns) == s.feature_names
    assert "last_txn_epoch" not in s.feature_names  # absolute time must not leak in


def test_score_shape_and_alert_flag():
    df = frame()
    s = RiskScorer(HistGradientBoostingClassifier(max_iter=5), threshold=0.5).fit(df, np.arange(len(df)) % 2)
    out = s.score(df)
    assert list(out.columns) == ["chargeback_risk", "alert"]
    assert ((out["chargeback_risk"] >= 0.5) == out["alert"]).all()


def test_threshold_respects_capacity_with_ties():
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
    t = threshold_for_capacity(scores, n_days=1, daily_capacity=2)
    assert (scores >= t).sum() <= 2
    assert threshold_for_capacity(scores, n_days=1, daily_capacity=10) == 0.1


def test_evaluate_counts():
    m = evaluate([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], threshold=0.5, n_days=2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["alerts_per_day"] == 1.0


def test_psi_zero_on_same_distribution_and_flags_shift():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"amt": rng.lognormal(3, 1, 20000), "email": rng.choice(["g", "y"], 20000)})
    base = drift.build_baseline(train, ["amt"], ["email"])
    same = drift.feature_psi(base, train.sample(5000, random_state=2))
    assert (same["psi"] < 0.01).all()

    shifted = pd.DataFrame({"amt": rng.lognormal(3.6, 1, 5000), "email": rng.choice(["g", "new"], 5000)})
    res = drift.feature_psi(base, shifted).set_index("feature")
    assert res.loc["amt", "status"] == "alert"
    assert res.loc["email", "status"] == "alert"  # unseen category lands in __other__


def test_missing_values_spike_is_drift():
    train = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    base = drift.build_baseline(train, ["x"], [])
    broken = pd.DataFrame({"x": [np.nan] * 600 + list(range(400))})
    res = drift.feature_psi(base, broken)
    assert res.loc[0, "status"] == "alert" and res.loc[0, "missing_rate"] == 0.6


def test_numeric_bin_rule_matches_spark_rule():
    edges = [1.0, 2.0, 3.0]
    v = pd.Series([0.5, 1.0, 1.5, 3.0, 9.0, np.nan])
    numpy_bins = drift.numeric_bins(v, edges).tolist()
    spark_rule = [str(sum(x > e for e in edges)) if not np.isnan(x) else drift.MISSING for x in v]
    assert numpy_bins == spark_rule

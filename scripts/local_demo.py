"""End-to-end dry run of the pipeline logic on synthetic data, no Spark or Databricks.

Same feature definitions, point-in-time lookup, scorer, threshold rule and PSI as the
Databricks path. Uses scikit-learn's HistGradientBoostingClassifier when xgboost is not
installed. Metrics here describe the planted synthetic pattern, not IEEE-CIS.

    python scripts/local_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from disputes_ml import drift  # noqa: E402
from disputes_ml.config import DAY, REF_EPOCH, TRAIN_END_DAY, VALID_END_DAY  # noqa: E402
from disputes_ml.features_ref import (  # noqa: E402
    asof_lookup_pd, compute_card_features_pd, incremental_update_pd, prepare_pd)
from disputes_ml.metrics import evaluate, threshold_for_capacity  # noqa: E402
from disputes_ml.schema import CARD_FEATURES, CATEGORICAL, LABEL, NUMERIC_REQUEST  # noqa: E402
from disputes_ml.scorer import RiskScorer  # noqa: E402
from disputes_ml.synthetic import generate  # noqa: E402


def estimator():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                 tree_method="hist"), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              random_state=0), "sklearn-hgb (stand-in)"


def main() -> dict:
    raw = generate(n_cards=2500, seed=11)
    tx = prepare_pd(raw)
    tx["event_day"] = tx["TransactionDT"] // DAY
    hist = tx[tx["event_day"] < VALID_END_DAY]
    live = tx[tx["event_day"] >= VALID_END_DAY]

    # batch features on history, point-in-time lookup at event - 1s
    feats = compute_card_features_pd(hist)
    spine = hist.assign(lookup_epoch=hist["event_epoch"] - 1)
    data = asof_lookup_pd(spine, feats, "lookup_epoch")
    tr = data[data["event_day"] < TRAIN_END_DAY]
    va = data[data["event_day"] >= TRAIN_END_DAY]

    est, est_name = estimator()
    scorer = RiskScorer(est).fit(tr, tr[LABEL])
    p_va = scorer.predict_proba(va)
    n_days = VALID_END_DAY - TRAIN_END_DAY
    capacity = max(1, int(round(len(va) / n_days * 0.05)))  # analysts review ~5% of traffic
    scorer.threshold = threshold_for_capacity(p_va, n_days, capacity)
    valid = evaluate(va[LABEL], p_va, scorer.threshold, n_days)

    # "stream" the live days: score with state before the day, then update state
    table, history, scored = feats.copy(), hist.copy(), []
    for day, batch in live.groupby("event_day"):
        req = asof_lookup_pd(batch.assign(lookup_epoch=batch["event_epoch"] - 1),
                             table, "lookup_epoch")
        # online semantics: latest row only, already <= event time because state is pre-day
        scored.append(req.assign(chargeback_risk=scorer.predict_proba(req)))
        history = pd.concat([history, batch], ignore_index=True)
        table = incremental_update_pd(table, history, batch)
    scored = pd.concat(scored, ignore_index=True)
    stream_ok = table.equals(compute_card_features_pd(history)) or np.allclose(
        table[CARD_FEATURES].astype(float).fillna(-1),
        compute_card_features_pd(history)[CARD_FEATURES].astype(float).fillna(-1))

    live_days = live["event_day"].nunique()
    live_perf = evaluate(scored[LABEL], scored["chargeback_risk"], scorer.threshold, live_days)

    # drift: request fields + score from the scored rows; card features from feature-table
    # rows (what the online store serves), validation window vs live window
    base = drift.build_baseline(va.assign(chargeback_risk=p_va),
                                NUMERIC_REQUEST + ["chargeback_risk"], CATEGORICAL)
    card_cols = [c for c in CARD_FEATURES if c != "last_txn_epoch"]
    row_day = (table["feature_epoch"] - REF_EPOCH) // DAY
    card_base = drift.build_baseline(
        table[(row_day >= TRAIN_END_DAY) & (row_day < VALID_END_DAY)], card_cols, [])
    psi = pd.concat([
        drift.feature_psi(base, scored),
        drift.feature_psi(card_base, table[row_day >= VALID_END_DAY]),
    ]).sort_values("psi", ascending=False)

    report = {
        "estimator": est_name,
        "rows": {"history": len(hist), "live": len(live), "train": len(tr), "valid": len(va)},
        "valid": {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in valid.items()},
        "live": {k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in live_perf.items()},
        "stream_equals_full_recompute": bool(stream_ok),
        "top_drift": psi.head(6).round(3).to_dict("records"),
        "capacity_per_day": capacity,
    }
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))

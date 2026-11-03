"""Model-side preprocessing + scoring, shared by training, the MLflow pyfunc and tests.

Everything that can be computed from the request plus looked-up features is computed
here, so offline training and online serving run the same code (no training-serving skew).
The estimator is anything with fit / predict_proba (XGBoost on Databricks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import CARD_FEATURES, CATEGORICAL, NUMERIC_REQUEST, ON_DEMAND

MISSING_CODE = -2
OTHER_CODE = -1


def add_on_demand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amt = pd.to_numeric(out["TransactionAmt"], errors="coerce").astype(float)
    last = pd.to_numeric(out["last_txn_epoch"], errors="coerce").astype(float)
    now = pd.to_numeric(out["event_epoch"], errors="coerce").astype(float)
    mean7 = pd.to_numeric(out["amt_mean_7d"], errors="coerce").astype(float)
    out["secs_since_last_txn"] = now - last          # NaN for a card's first transaction
    out["amt_log"] = np.log1p(amt)
    out["amt_cents"] = np.round(amt - np.floor(amt), 3)
    out["amt_vs_mean_7d"] = amt / mean7.replace(0, np.nan)
    return out


# The estimator sees these columns, in this order. last_txn_epoch is an absolute time, so
# it is only used to derive secs_since_last_txn, never fed to the model directly.
MODEL_NUMERIC = NUMERIC_REQUEST + [c for c in CARD_FEATURES if c != "last_txn_epoch"] + ON_DEMAND


@dataclass
class RiskScorer:
    estimator: object
    threshold: float = 0.5
    top_k: int = 50
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return MODEL_NUMERIC + CATEGORICAL

    def fit_vocab(self, df: pd.DataFrame) -> "RiskScorer":
        """Category -> integer code, from the training period only (top_k most frequent)."""
        self.vocab = {}
        for c in CATEGORICAL:
            top = df[c].dropna().astype(str).value_counts().head(self.top_k).index
            self.vocab[c] = {v: i for i, v in enumerate(sorted(top))}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = add_on_demand(df)
        out = pd.DataFrame(index=x.index)
        for c in MODEL_NUMERIC:
            out[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
        for c in CATEGORICAL:
            mapping = self.vocab.get(c, {})
            col = x[c]
            codes = col.astype(object).map(lambda v: mapping.get(str(v), OTHER_CODE))
            out[c] = np.where(col.isna(), MISSING_CODE, codes).astype(float)
        return out[self.feature_names]

    def fit(self, df: pd.DataFrame, y) -> "RiskScorer":
        self.fit_vocab(df)
        self.estimator.fit(self.transform(df), np.asarray(y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.transform(df))[:, 1]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(df)
        return pd.DataFrame({"chargeback_risk": p, "alert": p >= self.threshold}, index=df.index)

"""Population Stability Index for numeric and categorical features (numpy reference).

Binning rule, mirrored in Spark by `monitoring.spark_bin_index`:
    bin = number of inner edges strictly below the value   (np.searchsorted side="left")
    missing values get their own bin, so a jump in nulls shows up as drift
    (the failure mode behind the Lloyds join incident).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

PSI_WARN = 0.10
PSI_ALERT = 0.25
EPS = 1e-4
MISSING = "__missing__"
OTHER = "__other__"


def fit_numeric_edges(values, n_bins: int = 10) -> list[float]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return []
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    return sorted(set(float(q) for q in qs))


def numeric_bins(values, edges: list[float]) -> pd.Series:
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    idx = np.searchsorted(np.asarray(edges, float), v.fillna(0).to_numpy(float), side="left")
    labels = pd.Series(idx.astype(str), index=v.index)
    labels[v.isna()] = MISSING
    return labels


def fit_categories(values, top_k: int = 20) -> list[str]:
    s = pd.Series(values).dropna().astype(str)
    return sorted(s.value_counts().head(top_k).index)


def categorical_bins(values, categories: list[str]) -> pd.Series:
    s = pd.Series(values)
    lab = s.astype(str).where(s.astype(str).isin(categories), OTHER)
    lab[s.isna()] = MISSING
    return lab


def proportions(labels: pd.Series, support: list[str]) -> np.ndarray:
    counts = labels.value_counts()
    total = max(len(labels), 1)
    return np.array([counts.get(k, 0) / total for k in support], dtype=float)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if np.isnan(value):
        return "no_data"
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def build_baseline(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Bin definitions + expected proportions from the training window."""
    base = {}
    for c in numeric:
        edges = fit_numeric_edges(df[c])
        support = [str(i) for i in range(len(edges) + 1)] + [MISSING]
        base[c] = {"kind": "numeric", "edges": edges, "support": support,
                   "expected": proportions(numeric_bins(df[c], edges), support).tolist()}
    for c in categorical:
        cats = fit_categories(df[c])
        support = cats + [OTHER, MISSING]
        base[c] = {"kind": "categorical", "categories": cats, "support": support,
                   "expected": proportions(categorical_bins(df[c], cats), support).tolist()}
    return base


def feature_psi(baseline: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, b in baseline.items():
        if b["kind"] == "numeric":
            lab = numeric_bins(df[c], b["edges"])
        else:
            lab = categorical_bins(df[c], b["categories"])
        actual = proportions(lab, b["support"])
        value = psi(np.array(b["expected"]), actual) if len(df) else float("nan")
        rows.append({"feature": c, "psi": value, "status": status(value), "n": len(df),
                     "missing_rate": float((lab == MISSING).mean()) if len(df) else float("nan")})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def dumps(baseline: dict) -> str:
    return json.dumps(baseline, sort_keys=True)

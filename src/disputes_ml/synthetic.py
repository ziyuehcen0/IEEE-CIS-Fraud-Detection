"""Synthetic transactions with the IEEE-CIS column layout.

For tests, the local demo and a smoke run on Databricks before the Kaggle data is in place.
It is NOT a stand-in for real results: the fraud pattern is planted (bursts of higher-value
transactions on compromised cards), and drift is planted after `drift_from_day`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAY

EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "outlook.com"]
FRAUD_EMAILS = ["anonymous.com", "protonmail.com", "mail.com"]
PRODUCTS = ["W", "C", "R", "H", "S"]


def generate(
    n_cards: int = 1500,
    days: int = 182,
    fraud_card_rate: float = 0.04,
    drift_from_day: int | None = 150,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for card in range(n_cards):
        card1 = 1000 + card
        addr1 = float(rng.integers(100, 540)) if rng.random() > 0.1 else np.nan
        start_day = int(rng.integers(-400, 0))
        email = rng.choice(EMAILS, p=[0.45, 0.2, 0.15, 0.08, 0.07, 0.05])
        card4 = rng.choice(["visa", "mastercard", "american express", "discover"], p=[.6, .3, .07, .03])
        card6 = rng.choice(["debit", "credit"], p=[.7, .3])
        mu = rng.normal(3.8, 0.6)
        n = int(rng.poisson(rng.uniform(5, 60)))
        times = np.sort(rng.integers(0, days * DAY, size=n))
        compromised_at = (
            int(rng.integers(10 * DAY, days * DAY)) if rng.random() < fraud_card_rate else None
        )
        for t in times:
            rows.append(_row(rng, card1, addr1, start_day, email, card4, card6, mu, int(t),
                             fraud=False, drift_from_day=drift_from_day))
        if compromised_at is not None:
            burst = compromised_at + np.cumsum(rng.integers(30, 1800, size=int(rng.integers(3, 12))))
            for t in burst[burst < days * DAY]:
                rows.append(_row(rng, card1, addr1, start_day, rng.choice(FRAUD_EMAILS), card4,
                                 card6, mu + 1.2, int(t), fraud=True, drift_from_day=drift_from_day))
    df = pd.DataFrame(rows).sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(2_987_000, 2_987_000 + len(df), dtype=np.int64))
    return df


def _row(rng, card1, addr1, start_day, email, card4, card6, mu, t, fraud, drift_from_day):
    day = t // DAY
    drifted = drift_from_day is not None and day >= drift_from_day
    amt = float(np.round(np.exp(rng.normal(mu + (0.3 if drifted else 0.0), 0.7)), 2))
    p_email = "newpay.com" if drifted and rng.random() < 0.15 else email
    return {
        "TransactionDT": int(t) + 86_400,  # IEEE-CIS starts at 86400
        "isFraud": int(fraud),
        "TransactionAmt": amt,
        "ProductCD": rng.choice(PRODUCTS, p=[.7, .1, .08, .07, .05]) if not fraud
        else rng.choice(["C", "W"], p=[.6, .4]),
        "card1": card1,
        "card4": card4,
        "card6": card6,
        "addr1": addr1,
        "dist1": float(rng.exponential(30)) if rng.random() > 0.5 else np.nan,
        "P_emaildomain": p_email if rng.random() > 0.1 else None,
        "R_emaildomain": rng.choice(EMAILS) if rng.random() < 0.25 else None,
        "C1": float(rng.poisson(3 if fraud else 1.5)), "C2": float(rng.poisson(1.5)),
        "C5": float(rng.poisson(0.5)), "C6": float(rng.poisson(1.2)),
        "C13": float(rng.poisson(8)), "C14": float(rng.poisson(1.3)),
        "D1": float((t + 86_400) // DAY - start_day),
        "D4": float(rng.integers(0, 400)) if rng.random() > 0.3 else np.nan,
        "D10": float(rng.integers(0, 400)) if rng.random() > 0.2 else np.nan,
        "D15": float(rng.integers(0, 400)) if rng.random() > 0.2 else np.nan,
        "M4": rng.choice(["M0", "M1", "M2", None], p=[.3, .1, .2, .4]) if not fraud
        else rng.choice(["M2", None], p=[.6, .4]),
        "M6": rng.choice(["T", "F", None], p=[.4, .4, .2]),
    }

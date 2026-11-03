"""Column contracts shared by batch, streaming, training and serving."""

from __future__ import annotations

# Raw IEEE-CIS train_transaction columns we use, with the types we cast to.
RAW_TYPES: dict[str, str] = {
    "TransactionID": "bigint",
    "TransactionDT": "bigint",
    "TransactionAmt": "double",
    "ProductCD": "string",
    "card1": "int",
    "card4": "string",
    "card6": "string",
    "addr1": "double",
    "dist1": "double",
    "P_emaildomain": "string",
    "R_emaildomain": "string",
    "C1": "double", "C2": "double", "C5": "double", "C6": "double",
    "C13": "double", "C14": "double",
    "D1": "double", "D4": "double", "D10": "double", "D15": "double",
    "M4": "string", "M6": "string",
}
LABEL = "isFraud"  # 1 = transaction later received a chargeback (Vesta's definition)

# Landing-zone schema for the stream (no label: labels arrive later, separately).
LANDING_DDL = ", ".join(f"{c} {t}" for c, t in RAW_TYPES.items())

# Sent in every scoring request (known at authorisation time).
CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M6"]
NUMERIC_REQUEST = [
    "TransactionAmt", "dist1",
    "C1", "C2", "C5", "C6", "C13", "C14",
    "D1", "D4", "D10", "D15",
]
REQUEST_COLS = NUMERIC_REQUEST + CATEGORICAL + ["event_epoch"]

# Looked up from the feature store by card_uid.
CARD_FEATURES = [
    "txn_cnt_1h", "txn_cnt_24h", "txn_cnt_7d",
    "amt_sum_24h", "amt_mean_7d", "amt_std_7d", "amt_max_7d",
    "n_email_7d", "n_product_7d",
    "last_txn_epoch",
]

# Computed inside the model at request time (identical offline and online).
ON_DEMAND = ["secs_since_last_txn", "amt_log", "amt_cents", "amt_vs_mean_7d"]

KEY = "card_uid"
FEATURE_TS = "feature_ts"

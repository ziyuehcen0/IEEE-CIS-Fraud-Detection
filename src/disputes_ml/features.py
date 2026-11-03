"""Card-level rolling features in PySpark. Spec: `features_ref.compute_card_features_pd`.

Each output row is the card's state *after* every transaction at `feature_ts`, over the
window (feature_ts - W, feature_ts]. Training looks features up at event_ts - 1 second,
so a transaction only ever sees earlier transactions; serving reads the latest row,
which is the state after the card's most recent transaction. Same semantics both ways.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import WINDOWS
from .schema import CARD_FEATURES, FEATURE_TS, KEY

FEATURE_DDL = """
  card_uid STRING NOT NULL,
  feature_ts TIMESTAMP NOT NULL,
  txn_cnt_1h BIGINT,
  txn_cnt_24h BIGINT,
  txn_cnt_7d BIGINT,
  amt_sum_24h DOUBLE,
  amt_mean_7d DOUBLE,
  amt_std_7d DOUBLE,
  amt_max_7d DOUBLE,
  n_email_7d INT,
  n_product_7d INT,
  last_txn_epoch BIGINT
"""


def compute_card_features(tx: DataFrame) -> DataFrame:
    """tx needs: card_uid, event_ts, TransactionAmt, P_emaildomain, ProductCD."""
    t = F.col("event_ts").cast("long")
    base = Window.partitionBy(KEY).orderBy(t)

    def w(seconds: int):
        # integer-second timestamps: (t - W, t]  ==  [t - W + 1, t]
        return base.rangeBetween(-(seconds - 1), 0)

    h1, h24, d7 = (w(WINDOWS[k]) for k in ("1h", "24h", "7d"))
    amt = F.col("TransactionAmt").cast("double")

    out = tx.select(
        KEY,
        F.col("event_ts").alias(FEATURE_TS),
        F.count(F.lit(1)).over(h1).cast("bigint").alias("txn_cnt_1h"),
        F.count(F.lit(1)).over(h24).cast("bigint").alias("txn_cnt_24h"),
        F.count(F.lit(1)).over(d7).cast("bigint").alias("txn_cnt_7d"),
        F.sum(amt).over(h24).cast("double").alias("amt_sum_24h"),
        F.avg(amt).over(d7).cast("double").alias("amt_mean_7d"),
        F.stddev_samp(amt).over(d7).cast("double").alias("amt_std_7d"),
        F.max(amt).over(d7).cast("double").alias("amt_max_7d"),
        F.size(F.collect_set("P_emaildomain").over(d7)).cast("int").alias("n_email_7d"),
        F.size(F.collect_set("ProductCD").over(d7)).cast("int").alias("n_product_7d"),
        t.cast("bigint").alias("last_txn_epoch"),
    )
    # Rows sharing a second share a range frame, so their values are identical: keep one.
    return out.dropDuplicates([KEY, FEATURE_TS]).select(KEY, FEATURE_TS, *CARD_FEATURES)

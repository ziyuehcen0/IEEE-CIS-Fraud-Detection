"""Raw IEEE-CIS rows -> typed silver rows with card_uid and event time (PySpark).

Used by the batch ingest AND by every streaming micro-batch, so both paths produce the
same keys. Spec: `features_ref.prepare_pd` / `features_ref.card_uid`.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from .config import DAY, REF_EPOCH
from .schema import KEY, RAW_TYPES


def _int_str(c: Column) -> Column:
    return F.coalesce(c.cast("int").cast("string"), F.lit("NA"))


def prepare_transactions(raw: DataFrame) -> DataFrame:
    typed = raw.select(*[F.col(c).cast(t).alias(c) for c, t in RAW_TYPES.items()])
    day = F.floor(F.col("TransactionDT") / F.lit(DAY))
    uid = F.concat_ws(
        "_",
        F.col("card1").cast("string"),
        _int_str(F.col("addr1")),
        _int_str(day - F.col("D1")),
    )
    epoch = (F.lit(REF_EPOCH) + F.col("TransactionDT")).cast("bigint")
    return (
        typed.withColumn(KEY, uid)
        .withColumn("event_epoch", epoch)
        .withColumn("event_ts", F.timestamp_seconds(epoch))
        .withColumn("event_day", day.cast("int"))
    )

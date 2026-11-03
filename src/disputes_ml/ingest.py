"""Raw CSV (or synthetic) -> silver transactions, chargeback labels and the held-back live set."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import DAY, REF_EPOCH, VALID_END_DAY, Config
from .prep import prepare_transactions
from .schema import LABEL, RAW_TYPES


def read_raw(spark: SparkSession, cfg: Config, source: str) -> DataFrame:
    if source == "synthetic":
        from .synthetic import generate

        pdf = generate()
        for c in pdf.columns:  # object columns with None -> string, so Arrow infers cleanly
            if pdf[c].dtype == object:
                pdf[c] = pdf[c].astype("string")
        return spark.createDataFrame(pdf)
    path = f"{cfg.volume_path}/ieee-cis/train_transaction.csv"
    return spark.read.option("header", True).csv(path)


def ingest(spark: SparkSession, cfg: Config, source: str = "kaggle") -> dict:
    raw = read_raw(spark, cfg, source)
    raw = raw.withColumn("_day", F.floor(F.col("TransactionDT").cast("bigint") / DAY).cast("int"))

    history = prepare_transactions(raw.where(F.col("_day") < VALID_END_DAY))
    (history.write.mode("overwrite").option("overwriteSchema", True)
     .clusterBy("card_uid").saveAsTable(cfg.transactions))

    # Labels for every row, with the time they would have become known. IEEE-CIS gives no
    # chargeback date, so maturity is an assumption (Config.label_maturity_days).
    epoch = F.lit(REF_EPOCH) + F.col("TransactionDT").cast("bigint")
    labels = raw.select(
        F.col("TransactionID").cast("bigint").alias("TransactionID"),
        F.col(LABEL).cast("int").alias(LABEL),
        F.timestamp_seconds(epoch).alias("event_ts"),
        F.timestamp_seconds(epoch + F.lit(cfg.label_maturity_days * DAY)).alias("label_available_ts"),
    )
    labels.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(cfg.labels)

    # The last days are not in silver: they arrive later through the stream.
    live = raw.where(F.col("_day") >= VALID_END_DAY).select(*RAW_TYPES.keys(), "_day")
    live.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(cfg.live_source)

    return {
        "silver_rows": spark.table(cfg.transactions).count(),
        "live_rows": spark.table(cfg.live_source).count(),
        "label_rows": spark.table(cfg.labels).count(),
    }

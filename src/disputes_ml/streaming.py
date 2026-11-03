"""Structured Streaming: new transactions -> silver -> incremental feature upserts.

Spec of the per-batch logic: `features_ref.incremental_update_pd` (unit-tested to equal a
full recompute, including out-of-order arrivals).

The feature table has Change Data Feed on, so a CONTINUOUS / TRIGGERED online-store
publish picks up each MERGE and the serving endpoint sees new card state.
"""

from __future__ import annotations

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import MAX_LOOKBACK, Config
from .features import compute_card_features
from .prep import prepare_transactions
from .schema import FEATURE_TS, KEY, LANDING_DDL


def make_batch_processor(spark: SparkSession, cfg: Config):
    def process(batch_df: DataFrame, batch_id: int) -> None:
        batch = prepare_transactions(batch_df).dropDuplicates(["TransactionID"])
        stats = batch.agg(F.count("*").alias("n"), F.min("event_epoch").alias("min_epoch")).first()
        if stats["n"] == 0:
            return
        min_epoch = int(stats["min_epoch"])

        # 1. Append to silver; MERGE makes a replayed micro-batch idempotent.
        (
            DeltaTable.forName(spark, cfg.transactions).alias("s")
            .merge(batch.alias("b"), "s.TransactionID = b.TransactionID")
            .whenNotMatchedInsertAll()
            .execute()
        )

        # 2. Recompute affected cards from the batch's earliest event, with enough lookback.
        cards = batch.select(KEY).distinct()
        history = (
            spark.table(cfg.transactions)
            .where(F.col("event_epoch") > F.lit(min_epoch - MAX_LOOKBACK))
            .join(F.broadcast(cards), KEY)
        )
        feats = compute_card_features(history).where(
            F.col(FEATURE_TS).cast("long") >= F.lit(min_epoch)
        )

        # 3. Upsert into the offline feature table (primary key: card_uid, feature_ts).
        (
            DeltaTable.forName(spark, cfg.card_features).alias("f")
            .merge(feats.alias("n"), f"f.{KEY} = n.{KEY} AND f.{FEATURE_TS} = n.{FEATURE_TS}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    return process


def start_feature_stream(spark: SparkSession, cfg: Config, serverless: bool = True):
    """Auto Loader on the landing volume -> foreachBatch.

    Serverless compute (incl. Free Edition) only supports availableNow triggers, so there
    the job is scheduled every few minutes. On classic compute a processingTime trigger
    gives a continuously running query.
    """
    trigger = {"availableNow": True} if serverless else {"processingTime": "30 seconds"}
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", cfg.checkpoint_path + "/_schema")
        .schema(LANDING_DDL)
        .load(cfg.landing_path)
    )
    return (
        stream.writeStream.foreachBatch(make_batch_processor(spark, cfg))
        .option("checkpointLocation", cfg.checkpoint_path)
        .trigger(**trigger)
        .queryName("card_features_stream")
        .start()
    )

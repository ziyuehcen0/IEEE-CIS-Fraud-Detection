"""Unity Catalog feature table (offline) + Databricks Online Feature Store (online)."""

from __future__ import annotations

import time

from pyspark.sql import SparkSession

from .config import Config
from .features import FEATURE_DDL, compute_card_features
from .schema import CARD_FEATURES, FEATURE_TS, KEY


def create_feature_table(spark: SparkSession, cfg: Config) -> None:
    """Time-series feature table: PK (card_uid, feature_ts TIMESERIES), CDF on for publishing."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {cfg.card_features} (
          {FEATURE_DDL},
          CONSTRAINT card_features_pk PRIMARY KEY ({KEY}, {FEATURE_TS} TIMESERIES)
        )
        CLUSTER BY ({KEY})
        COMMENT 'Card-level rolling features; row = card state after all txns at feature_ts'
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)


def backfill_features(spark: SparkSession, cfg: Config) -> int:
    feats = compute_card_features(spark.table(cfg.transactions))
    feats.select(KEY, FEATURE_TS, *CARD_FEATURES).createOrReplaceTempView("_card_features_backfill")
    spark.sql(f"INSERT OVERWRITE {cfg.card_features} SELECT * FROM _card_features_backfill")
    return spark.table(cfg.card_features).count()


def ensure_online_store(fe, cfg: Config, capacity: str = "CU_1", timeout_s: int = 1200):
    try:
        store = fe.get_online_store(name=cfg.online_store)
    except Exception:  # not found raises on some client versions
        store = None
    if not store:
        fe.create_online_store(name=cfg.online_store, capacity=capacity)
    deadline = time.time() + timeout_s
    while True:
        store = fe.get_online_store(name=cfg.online_store)
        if store and str(store.state).upper().endswith("AVAILABLE"):
            return store
        if time.time() > deadline:
            raise TimeoutError(f"online store {cfg.online_store} not AVAILABLE: {store and store.state}")
        time.sleep(20)


def publish_online(fe, cfg: Config, mode: str = "CONTINUOUS"):
    """Latest row per card goes online (time-series table => deduplicated by card_uid).

    CONTINUOUS keeps a streaming sync running (needs always-on compute);
    TRIGGERED syncs when called / scheduled, which is cheaper for a demo.
    """
    store = ensure_online_store(fe, cfg)
    return fe.publish_table(
        online_store=store,
        source_table_name=cfg.card_features,
        online_table_name=cfg.card_features_online,
        publish_mode=mode,
    )

# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 Parity check: Spark vs the unit-tested pandas reference
# MAGIC The PySpark feature code and the drift binning were written against a pandas reference
# MAGIC that has unit tests (`tests/`). This notebook proves they agree on real data, and runs the
# MAGIC streaming foreachBatch logic on a scratch table to check incremental == full recompute.
# MAGIC Run it after 02 and before trusting any model metric.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from disputes_ml.config import Config, REF_EPOCH
from disputes_ml.features import compute_card_features
from disputes_ml.features_ref import compute_card_features_pd
from disputes_ml.schema import CARD_FEATURES

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------


# Collect the sample once: a lazy rand()+limit is re-evaluated by every action and can
# return a different set of cards each time.
card_ids = [r.card_uid for r in (spark.table(cfg.transactions).groupBy("card_uid").count()
            .where("count between 5 and 400").orderBy(F.rand(42)).limit(300).collect())]
cards = spark.createDataFrame([(c,) for c in card_ids], "card_uid string")
tx = spark.table(cfg.transactions).join(cards.select("card_uid"), "card_uid")

spark_feats = (compute_card_features(tx)
               .withColumn("feature_epoch", F.col("feature_ts").cast("long"))
               .drop("feature_ts").toPandas())
ref = compute_card_features_pd(
    tx.select("card_uid", "event_epoch", "TransactionAmt", "P_emaildomain", "ProductCD").toPandas())

key = ["card_uid", "feature_epoch"]
a = spark_feats.sort_values(key).reset_index(drop=True)[key + CARD_FEATURES]
b = ref.sort_values(key).reset_index(drop=True)[key + CARD_FEATURES]
pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9, atol=1e-6)
print(f"OK: {len(a):,} feature rows identical across {cards.count()} cards")

# COMMAND ----------

# 2. The offline feature table equals a fresh recompute for those cards.
table = (spark.table(cfg.card_features).join(cards.select("card_uid"), "card_uid")
         .withColumn("feature_epoch", F.col("feature_ts").cast("long")).drop("feature_ts")
         .toPandas().sort_values(key).reset_index(drop=True)[key + CARD_FEATURES])
pd.testing.assert_frame_equal(table, b, check_dtype=False, rtol=1e-9, atol=1e-6)
print("OK: feature table matches reference")

# COMMAND ----------

# 3. Streaming foreachBatch on scratch tables: replay sampled history in shuffled batches.
from dataclasses import replace
from disputes_ml.feature_store import create_feature_table
from disputes_ml.schema import RAW_TYPES
from disputes_ml.streaming import make_batch_processor

scratch = replace(cfg, schema=cfg.schema + "_parity")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {scratch.catalog}.{scratch.schema}")
spark.sql(f"DROP TABLE IF EXISTS {scratch.card_features}")
create_feature_table(spark, scratch)
tx.limit(0).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(scratch.transactions)

raw_cols = list(RAW_TYPES)
batches = tx.select(*raw_cols).withColumn("_b", (F.rand(7) * 8).cast("int"))  # out of order
process = make_batch_processor(spark, scratch)
for i in range(8):
    process(batches.where(F.col("_b") == i).drop("_b"), i)
process(batches.where(F.col("_b") == 3).drop("_b"), 99)  # replayed batch must be idempotent

streamed = (spark.table(scratch.card_features)
            .withColumn("feature_epoch", F.col("feature_ts").cast("long")).drop("feature_ts")
            .toPandas().sort_values(key).reset_index(drop=True)[key + CARD_FEATURES])
pd.testing.assert_frame_equal(streamed, b, check_dtype=False, rtol=1e-9, atol=1e-6)
print("OK: incremental streaming updates == full recompute (out-of-order + replayed batch)")
spark.sql(f"DROP SCHEMA {scratch.catalog}.{scratch.schema} CASCADE")

# COMMAND ----------

# 4. Drift binning: Spark expression == numpy reference.
from disputes_ml import drift
from disputes_ml.monitoring import spark_bin

pdf = (spark.table(cfg.transactions).select("TransactionAmt", "dist1", "P_emaildomain")
       .limit(50000).toPandas())
vals = spark.createDataFrame(pdf, "TransactionAmt double, dist1 double, P_emaildomain string")
for col, kind in [("TransactionAmt", "numeric"), ("dist1", "numeric"), ("P_emaildomain", "categorical")]:
    if kind == "numeric":
        bins = drift.fit_numeric_edges(pdf[col]); ref_lab = drift.numeric_bins(pdf[col], bins)
    else:
        bins = drift.fit_categories(pdf[col]); ref_lab = drift.categorical_bins(pdf[col], bins)
    got = vals.select(spark_bin(F.col(col), {"kind": kind, "bins": bins}).alias("b")).toPandas()["b"]
    assert (got.values == ref_lab.values).all(), col
print("OK: Spark drift bins == numpy bins")

# COMMAND ----------

from pyspark.sql import functions as F

ft = spark.table(cfg.card_features)
print("feature table rows:", ft.count(), "| silver rows:", spark.table(cfg.transactions).count())
print("duplicate PK groups:", ft.groupBy("card_uid", "feature_ts").count().where("count > 1").count())

extra = table.merge(b[key], on=key, how="left", indicator=True).query("_merge == 'left_only'")
print("extra rows:", len(extra))
display(extra.head(10))

display(spark.sql(f"DESCRIBE HISTORY {cfg.card_features}")
        .select("version", "timestamp", "operation").limit(5))
# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Structured Streaming: landing files -> silver -> feature table upserts
# MAGIC Auto Loader reads JSON dropped by the replay (07). Each micro-batch is appended to silver
# MAGIC (MERGE on TransactionID, so replays are idempotent), affected cards are recomputed from
# MAGIC the batch's earliest event with a 7-day lookback, and rows are MERGEd into the feature
# MAGIC table. Change Data Feed carries them to the online store.
# MAGIC
# MAGIC Serverless (incl. Free Edition) supports only `availableNow`; schedule this as a job every
# MAGIC few minutes. On classic compute set `serverless=false` for a 30-second processing trigger.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))
from disputes_ml.config import Config
from disputes_ml.streaming import start_feature_stream

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.text("serverless", "true")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

q = start_feature_stream(spark, cfg, serverless=dbutils.widgets.get("serverless") == "true")
q.awaitTermination() if dbutils.widgets.get("serverless") == "true" else None
print(q.lastProgress)

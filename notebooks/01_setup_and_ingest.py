# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 Setup and ingest
# MAGIC Creates the schema and volume, then splits IEEE-CIS `train_transaction.csv` into
# MAGIC - `transactions` (silver, days 0-149, no label),
# MAGIC - `chargeback_labels` (label + simulated time it becomes known),
# MAGIC - `live_source` (days 150+, replayed later through the stream).
# MAGIC
# MAGIC **Data:** download from Kaggle (competition `ieee-fraud-detection`, accept the rules) and upload
# MAGIC `train_transaction.csv` to `/Volumes/<catalog>/<schema>/raw/ieee-cis/`. Set `source=synthetic`
# MAGIC for a smoke run without it.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))
from disputes_ml.config import Config

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.dropdown("source", "kaggle", ["kaggle", "synthetic"])
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{cfg.volume}")
dbutils.fs.mkdirs(cfg.landing_path)

# COMMAND ----------

from disputes_ml.ingest import ingest

counts = ingest(spark, cfg, source=dbutils.widgets.get("source"))
print(counts)

# COMMAND ----------

display(spark.sql(f"""
  SELECT event_day DIV 30 AS month, count(*) AS txns, round(avg(l.isFraud), 4) AS chargeback_rate
  FROM {cfg.transactions} t JOIN {cfg.labels} l USING (TransactionID)
  GROUP BY 1 ORDER BY 1
"""))
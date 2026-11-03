# Databricks notebook source
# MAGIC %md
# MAGIC # 04 Train XGBoost, track in MLflow, register in Unity Catalog
# MAGIC - Point-in-time training set: `FeatureLookup(timestamp_lookup_key="lookup_ts")`, with
# MAGIC   `lookup_ts = event_ts - 1s`, so a transaction never sees its own card update.
# MAGIC - Out-of-time split: train days [0,120), validation [120,150), live replay 150+.
# MAGIC - Threshold set on validation so alerts/day stay within analyst capacity.
# MAGIC - Model logged with `fe.log_model` (feature spec travels with it -> automatic online lookup).
# MAGIC - Promotion gate: `champion` alias only if validation PR-AUC is not worse.
# MAGIC
# MAGIC **Label caveat:** IEEE-CIS `isFraud` = the transaction later had a chargeback. This is a
# MAGIC chargeback-risk model, not a dispute win/loss model. The split is retrospective: in
# MAGIC production, training data would stop `label_maturity_days` before today.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, json
sys.path.append(os.path.abspath("../src"))
from databricks.feature_engineering import FeatureEngineeringClient
from disputes_ml.config import Config
from disputes_ml.train import train_and_register

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.text("use_online_store", "true")
dbutils.widgets.text("daily_alert_capacity", "400")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"),
             daily_alert_capacity=int(dbutils.widgets.get("daily_alert_capacity")))
USE_ONLINE = dbutils.widgets.get("use_online_store").lower() == "true"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

result = train_and_register(spark, FeatureEngineeringClient(), cfg, use_online_store=USE_ONLINE)
print(json.dumps(result, indent=2, default=str))
dbutils.jobs.taskValues.set("model_version", result["version"])

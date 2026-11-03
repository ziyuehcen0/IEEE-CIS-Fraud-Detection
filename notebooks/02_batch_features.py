# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 Batch features -> feature store (offline + online)
# MAGIC PySpark window features per card over (t - W, t]; written to a Unity Catalog time-series
# MAGIC feature table (PK `card_uid, feature_ts TIMESERIES`, Change Data Feed on), then published
# MAGIC to a Databricks Online Feature Store (Lakebase). Online keeps the latest row per card.

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
dbutils.widgets.text("use_online_store", "true")
dbutils.widgets.dropdown("publish_mode", "CONTINUOUS", ["CONTINUOUS", "TRIGGERED"])
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
USE_ONLINE = dbutils.widgets.get("use_online_store").lower() == "true"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

from disputes_ml.feature_store import backfill_features, create_feature_table

create_feature_table(spark, cfg)
print("feature rows:", backfill_features(spark, cfg))
display(spark.table(cfg.card_features).orderBy("card_uid", "feature_ts").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Online store: needs a workspace with Lakebase / Online Feature Store (not Free Edition).
# MAGIC With `use_online_store=false` the rest of the pipeline still runs; the replay then sends
# MAGIC card features in the request instead.

# COMMAND ----------

if USE_ONLINE:
    from databricks.feature_engineering import FeatureEngineeringClient
    from disputes_ml.feature_store import publish_online

    fe = FeatureEngineeringClient()
    pub = publish_online(fe, cfg, mode=dbutils.widgets.get("publish_mode"))
    print(pub)
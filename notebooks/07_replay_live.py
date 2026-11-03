# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Replay the held-back days as live traffic
# MAGIC For each day: score that day's transactions on the endpoint (card state = everything
# MAGIC before the day), log request + response to `scoring_log`, then drop the transactions into
# MAGIC the landing zone and run the stream so card state moves forward.
# MAGIC
# MAGIC Simplification: card state advances once per replayed day, so two transactions on the
# MAGIC same card on the same day don't see each other here. Smaller chunks fix that at the cost
# MAGIC of more stream runs.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, time
sys.path.append(os.path.abspath("../src"))
from pyspark.sql import functions as F
from disputes_ml.config import Config
from disputes_ml.monitoring import replay_day
from disputes_ml.streaming import start_feature_stream

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.text("use_online_store", "true")
dbutils.widgets.text("max_days", "32")
dbutils.widgets.text("online_sync_wait_s", "60")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
USE_ONLINE = dbutils.widgets.get("use_online_store").lower() == "true"
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

days = [r[0] for r in spark.table(cfg.live_source).select("_day").distinct().orderBy("_day").collect()]
days = days[: int(dbutils.widgets.get("max_days"))]
for d in days:
    n = replay_day(spark, cfg, d, use_online_store=USE_ONLINE)
    start_feature_stream(spark, cfg, serverless=True).awaitTermination()
    if USE_ONLINE:
        time.sleep(int(dbutils.widgets.get("online_sync_wait_s")))  # CONTINUOUS sync lag
    print(f"day {d}: scored {n}")

# COMMAND ----------

display(spark.table(cfg.scoring_log).groupBy("event_day")
        .agg(F.count("*").alias("n"), F.avg("chargeback_risk").alias("mean_risk"),
             F.sum(F.col("alert").cast("int")).alias("alerts"))
        .orderBy("event_day"))

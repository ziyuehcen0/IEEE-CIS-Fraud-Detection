# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Drift and delayed-label performance (scheduled daily)
# MAGIC - PSI per feature per day vs the training-time baseline; missing values get their own bin,
# MAGIC   so a broken upstream join shows up as drift.
# MAGIC - The job fails (-> e-mail via job notifications) if any feature alerts (PSI >= 0.25) on 2+ days.
# MAGIC - Precision / recall / alerts per day per week, only on labels mature at run time.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))
from disputes_ml.config import Config
from disputes_ml.monitoring import daily_drift, delayed_label_performance, raise_on_alert

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.text("fail_on_alert", "true")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

perf = delayed_label_performance(spark, cfg)
display(perf)

# COMMAND ----------

res = daily_drift(spark, cfg)
display(res.pivot_table(index="feature", columns="day", values="psi").round(3))

# COMMAND ----------

if dbutils.widgets.get("fail_on_alert") == "true":
    raise_on_alert(res)

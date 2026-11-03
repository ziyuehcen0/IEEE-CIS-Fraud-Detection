# Databricks notebook source
# MAGIC %md
# MAGIC # 05 Deploy the champion to a real-time Model Serving endpoint
# MAGIC CPU, scale-to-zero, request/response logged to an inference table. With the online store,
# MAGIC a request needs only `card_uid`, `lookup_ts` and the authorisation fields; the endpoint
# MAGIC looks up card features itself.

# COMMAND ----------

# MAGIC %pip install -q xgboost "databricks-feature-engineering>=0.13.0" "databricks-sdk>=0.40" "mlflow>=2.16"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath("../src"))
import pandas as pd
from disputes_ml.config import Config
from disputes_ml.serving import deploy_endpoint, score

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "disputes_ml")
dbutils.widgets.text("inference_table", "true")
dbutils.widgets.text("use_online_store", "true")
cfg = Config(catalog=dbutils.widgets.get("catalog"), schema=dbutils.widgets.get("schema"))

# COMMAND ----------

version = deploy_endpoint(cfg, inference_table=dbutils.widgets.get("inference_table") == "true")
print("serving version", version)

# COMMAND ----------

# Smoke request: a known card from the feature table.
uid = spark.table(cfg.card_features).select("card_uid").first()[0]
req = pd.DataFrame([{
    "card_uid": uid, "lookup_ts": pd.Timestamp.utcnow().isoformat(),
    "TransactionAmt": 117.0, "dist1": None, "C1": 1.0, "C2": 1.0, "C5": 0.0, "C6": 1.0,
    "C13": 1.0, "C14": 1.0, "D1": 0.0, "D4": None, "D10": 0.0, "D15": 0.0,
    "ProductCD": "W", "card4": "visa", "card6": "debit", "P_emaildomain": "gmail.com",
    "R_emaildomain": None, "M4": None, "M6": "T", "event_epoch": 1527700000,
}])
if dbutils.widgets.get("use_online_store") != "true":  # fallback: send card features too
    from disputes_ml.schema import CARD_FEATURES
    latest = (spark.table(cfg.card_features).where(f"card_uid = '{uid}'")
              .orderBy("feature_ts", ascending=False).limit(1).toPandas())
    req = req.join(latest[CARD_FEATURES])
display(score(cfg, req))

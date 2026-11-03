"""Point-in-time training set from the feature store -> XGBoost -> MLflow (Unity Catalog)."""

from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from . import drift
from .config import DAY, REF_EPOCH, TRAIN_END_DAY, VALID_END_DAY, Config
from .metrics import evaluate, threshold_for_capacity
from .schema import CARD_FEATURES, CATEGORICAL, KEY, LABEL, NUMERIC_REQUEST, REQUEST_COLS
from .scorer import RiskScorer

XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=7, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
    tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100, n_jobs=-1,
)
DRIFT_CARD = [c for c in CARD_FEATURES if c != "last_txn_epoch"]


def spine(spark: SparkSession, cfg: Config, day_from: int, day_to: int):
    """One row per labelled transaction; features are looked up at event time - 1 second."""
    return (
        spark.table(cfg.transactions)
        .where((F.col("event_day") >= day_from) & (F.col("event_day") < day_to))
        .join(spark.table(cfg.labels).select("TransactionID", LABEL), "TransactionID")
        .withColumn("lookup_ts", F.timestamp_seconds(F.col("event_epoch") - 1))
        .select(KEY, "lookup_ts", *REQUEST_COLS, LABEL)
    )


def training_set(fe, spark, cfg: Config, day_from: int, day_to: int):
    from databricks.feature_engineering import FeatureLookup

    lookups = [FeatureLookup(
        table_name=cfg.card_features,
        lookup_key=KEY,
        timestamp_lookup_key="lookup_ts",
        feature_names=CARD_FEATURES,
    )]
    return fe.create_training_set(
        df=spine(spark, cfg, day_from, day_to), feature_lookups=lookups, label=LABEL,
    )


def _pkg_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def train_and_register(spark, fe, cfg: Config, use_online_store: bool = True) -> dict:
    import xgboost as xgb

    from .model_wrapper import ChargebackRiskModel

    mlflow.set_registry_uri("databricks-uc")
    train_set = training_set(fe, spark, cfg, 0, TRAIN_END_DAY)
    valid_set = training_set(fe, spark, cfg, TRAIN_END_DAY, VALID_END_DAY)
    tr = train_set.load_df().toPandas()
    va = valid_set.load_df().toPandas()

    scorer = RiskScorer(xgb.XGBClassifier(**XGB_PARAMS)).fit_vocab(tr)
    xtr, xva = scorer.transform(tr), scorer.transform(va)
    scorer.estimator.fit(xtr, tr[LABEL], eval_set=[(xva, va[LABEL])], verbose=False)

    p_va = scorer.predict_proba(va)
    n_valid_days = VALID_END_DAY - TRAIN_END_DAY
    scorer.threshold = threshold_for_capacity(p_va, n_valid_days, cfg.daily_alert_capacity)
    m_va = evaluate(va[LABEL], p_va, scorer.threshold, n_valid_days)

    # Drift baseline from the validation window (closest to live). Request fields and the
    # score come from the scored rows; card features from the feature-table rows themselves,
    # because that is what the online store serves and what the monitor will read.
    baseline = drift.build_baseline(
        va.assign(chargeback_risk=p_va), NUMERIC_REQUEST + ["chargeback_risk"], CATEGORICAL)
    fr = (spark.table(cfg.card_features)
          .where(F.col("feature_ts").cast("long") - F.lit(REF_EPOCH) >= F.lit(TRAIN_END_DAY * DAY))
          .where(F.col("feature_ts").cast("long") - F.lit(REF_EPOCH) < F.lit(VALID_END_DAY * DAY))
          .select(*DRIFT_CARD).toPandas())
    baseline.update(drift.build_baseline(fr, DRIFT_CARD, []))

    importance = (
        pd.Series(scorer.estimator.feature_importances_, index=scorer.feature_names)
        .sort_values(ascending=False)
    )

    with mlflow.start_run(run_name="xgb_chargeback_risk") as run:
        mlflow.log_params({k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"})
        mlflow.log_params({
            "best_iteration": scorer.estimator.best_iteration,
            "train_days": f"[0,{TRAIN_END_DAY})", "valid_days": f"[{TRAIN_END_DAY},{VALID_END_DAY})",
            "daily_alert_capacity": cfg.daily_alert_capacity,
            "train_rows": len(tr), "train_base_rate": float(tr[LABEL].mean()),
        })
        mlflow.log_metrics({f"valid_{k}": v for k, v in m_va.items()
                            if isinstance(v, (int, float)) and np.isfinite(v)})
        mlflow.log_dict(baseline, "monitoring/drift_baseline.json")
        mlflow.log_dict(importance.round(5).to_dict(), "explain/feature_importance.json")

        model = ChargebackRiskModel(scorer, version_tag=run.info.run_id[:8])
        reqs = [f"xgboost=={xgb.__version__}", f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}", "scikit-learn"]
        example = va.drop(columns=[LABEL]).head(5)
        if use_online_store:
            # Feature spec travels with the model -> Model Serving looks features up by card_uid.
            kw = dict(model=model, flavor=mlflow.pyfunc, training_set=train_set,
                      registered_model_name=cfg.uc_model, code_paths=[_pkg_dir()],
                      pip_requirements=reqs, infer_input_example=True)
            try:
                fe.log_model(artifact_path="model", **kw)
            except TypeError:  # newer clients follow MLflow 3 and call it `name`
                fe.log_model(name="model", **kw)
        else:
            # Free Edition fallback: no online store, the caller sends the card features.
            mlflow.pyfunc.log_model(
                artifact_path="model", python_model=model, registered_model_name=cfg.uc_model,
                code_paths=[_pkg_dir()], pip_requirements=reqs, input_example=example,
            )
        run_id = run.info.run_id

    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(
        f"name='{cfg.uc_model}' and run_id='{run_id}'"))
    alias = _promote(client, cfg, version, m_va["pr_auc"])
    _write_baseline_table(spark, cfg, baseline, version)
    return {"run_id": run_id, "version": version, "alias": alias, "valid": m_va,
            "top_features": importance.head(10).round(4).to_dict()}


def _promote(client: MlflowClient, cfg: Config, version: int, pr_auc: float) -> str:
    """Gate: new version becomes champion only if validation PR-AUC is not worse."""
    try:
        champ = client.get_model_version_by_alias(cfg.uc_model, "champion")
        champ_pr = client.get_run(champ.run_id).data.metrics.get("valid_pr_auc", -1)
    except Exception:
        champ_pr = -1.0
    alias = "champion" if pr_auc >= champ_pr else "challenger"
    client.set_registered_model_alias(cfg.uc_model, alias, version)
    return alias


def _write_baseline_table(spark, cfg: Config, baseline: dict, version: int) -> None:
    import json

    rows = [(f, b["kind"], json.dumps(b.get("edges", b.get("categories"))),
             json.dumps(b["support"]), json.dumps(b["expected"]), int(version))
            for f, b in baseline.items()]
    df = spark.createDataFrame(
        rows, "feature string, kind string, bins string, support string, expected string, model_version int")
    df.write.mode("overwrite").saveAsTable(cfg.drift_baseline)

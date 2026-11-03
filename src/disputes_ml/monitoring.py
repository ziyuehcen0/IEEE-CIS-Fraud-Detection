"""Live replay, daily drift (PSI) and delayed-label performance (PySpark + numpy).

Drift is computed on:
  * request fields and the score, from `scoring_log` (what the endpoint received / returned)
  * card features, from the feature-table rows written by the stream (what the online
    store serves)
against the baseline logged at training time. Binning in Spark uses the same rule as
`drift.numeric_bins`, so the numbers match the unit-tested reference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pyspark.sql import Column, SparkSession, Window
from pyspark.sql import functions as F

from . import drift
from .config import DAY, REF_EPOCH, VALID_END_DAY, Config
from .metrics import evaluate
from .prep import prepare_transactions
from .schema import CARD_FEATURES, FEATURE_TS, KEY, LABEL, RAW_TYPES, REQUEST_COLS


# --------------------------------------------------------------------------- replay
def _offline_features(spark: SparkSession, cfg: Config, req):
    """Free Edition fallback: as-of join card features from Delta (not real-time)."""
    f = spark.table(cfg.card_features)
    j = req.join(f, KEY, "left").where(
        F.col(FEATURE_TS).isNull() | (F.col(FEATURE_TS).cast("long") <= F.col("event_epoch") - 1))
    w = Window.partitionBy("TransactionID").orderBy(F.col(FEATURE_TS).desc_nulls_last())
    return j.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn", FEATURE_TS)


def replay_day(spark: SparkSession, cfg: Config, day: int, use_online_store: bool = True) -> int:
    """Score one day of held-back transactions, then drop them into the landing zone."""
    from .serving import score

    live = spark.table(cfg.live_source).where(F.col("_day") == day).drop("_day")
    req = prepare_transactions(live).withColumn("lookup_ts", F.timestamp_seconds("event_epoch"))
    cols = ["TransactionID", KEY, "lookup_ts", "event_ts", "event_day", *REQUEST_COLS]
    req = req.select(*cols)
    if not use_online_store:
        req = _offline_features(spark, cfg, req)
    pdf = req.toPandas()
    if pdf.empty:
        return 0

    send = pdf.drop(columns=["TransactionID", "event_ts", "event_day"])
    preds = score(cfg, send)
    logged = pdf[["TransactionID", KEY, "event_ts", "event_day", *REQUEST_COLS]].join(preds)
    logged["scored_at"] = datetime.now(timezone.utc)
    spark.createDataFrame(logged).write.mode("append").option("mergeSchema", True) \
        .saveAsTable(cfg.scoring_log)

    # After scoring (as at authorisation), the transactions update card state via the stream.
    # Cast first: the CSV source is all strings, and the typed JSON reader would null them.
    typed = live.select(*[F.col(c).cast(t).alias(c) for c, t in RAW_TYPES.items()])
    typed.write.mode("overwrite").json(f"{cfg.landing_path}/day_{day:03d}")
    return len(pdf)


# --------------------------------------------------------------------------- drift
def spark_bin(col: Column, entry: dict) -> Column:
    if entry["kind"] == "numeric":
        idx = sum((F.when(col > F.lit(e), 1).otherwise(0) for e in entry["bins"]), F.lit(0))
        return F.when(col.isNull() | F.isnan(col.cast("double")), F.lit(drift.MISSING)) \
            .otherwise(idx.cast("string"))
    cats = entry["bins"]
    return F.when(col.isNull(), F.lit(drift.MISSING)) \
        .when(col.cast("string").isin(cats), col.cast("string")) \
        .otherwise(F.lit(drift.OTHER))


def load_baseline(spark: SparkSession, cfg: Config) -> dict:
    out = {}
    for r in spark.table(cfg.drift_baseline).collect():
        out[r.feature] = {"kind": r.kind, "bins": json.loads(r.bins),
                          "support": json.loads(r.support), "expected": json.loads(r.expected),
                          "model_version": r.model_version}
    return out


def daily_drift(spark: SparkSession, cfg: Config) -> pd.DataFrame:
    base = load_baseline(spark, cfg)
    log = spark.table(cfg.scoring_log)
    card = (spark.table(cfg.card_features)
            .withColumn("event_day", F.floor((F.col(FEATURE_TS).cast("long") - REF_EPOCH) / DAY))
            .where(F.col("event_day") >= VALID_END_DAY))
    rows = []
    for feature, b in base.items():
        src = card if feature in CARD_FEATURES else log
        if feature not in src.columns:
            continue
        counts = (src.select(F.col("event_day").cast("int").alias("day"),
                             spark_bin(F.col(feature), b).alias("bin"))
                  .groupBy("day", "bin").count().toPandas())
        expected = np.array(b["expected"])
        for day, g in counts.groupby("day"):
            c = g.set_index("bin")["count"]
            n = int(c.sum())
            actual = np.array([c.get(k, 0) / n for k in b["support"]])
            value = drift.psi(expected, actual)
            rows.append({"day": int(day), "feature": feature, "psi": value,
                         "status": drift.status(value), "n": n,
                         "missing_rate": float(c.get(drift.MISSING, 0) / n),
                         "model_version": b["model_version"]})
    res = pd.DataFrame(rows)
    if not res.empty:
        spark.createDataFrame(res).write.mode("overwrite").saveAsTable(cfg.drift_metrics)
    return res


def raise_on_alert(res: pd.DataFrame, min_days: int = 2) -> None:
    """Fail the job (-> job e-mail notification) if a feature alerts on >= min_days days."""
    if res.empty:
        return
    bad = res[res["status"] == "alert"].groupby("feature")["day"].nunique()
    bad = bad[bad >= min_days]
    if len(bad):
        raise RuntimeError(f"PSI >= {drift.PSI_ALERT} on {min_days}+ days: {bad.to_dict()}")


# --------------------------------------------------------------------------- performance
def delayed_label_performance(spark: SparkSession, cfg: Config, as_of=None) -> pd.DataFrame:
    """Precision / recall / alert volume per week, only on labels mature at `as_of`."""
    as_of_col = F.current_timestamp() if as_of is None else F.lit(as_of).cast("timestamp")
    joined = (spark.table(cfg.scoring_log).alias("s")
              .join(spark.table(cfg.labels).alias("l"), "TransactionID")
              .where(F.col("l.label_available_ts") <= as_of_col)
              .select(F.col("s.event_day").alias("day"), "chargeback_risk", "alert", LABEL)
              .toPandas())
    if joined.empty:
        return joined
    joined["week"] = (joined["day"] - VALID_END_DAY) // 7
    rows = []
    for week, g in joined.groupby("week"):
        m = evaluate(g[LABEL], g["chargeback_risk"], float("nan"), n_days=g["day"].nunique(),
                     alert=g["alert"].astype(bool))
        rows.append({"week": int(week), **m})
    res = pd.DataFrame(rows)
    spark.createDataFrame(res).write.mode("overwrite").saveAsTable(cfg.performance_metrics)
    return res

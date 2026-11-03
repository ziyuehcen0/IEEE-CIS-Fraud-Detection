"""Chargeback (dispute) risk model on Databricks, built on the IEEE-CIS payments data.

Pure-Python modules (no Spark / Databricks imports at module level):
    config, schema, features_ref, metrics, drift, scorer, synthetic
Spark / Databricks modules (import only on a cluster):
    prep, features, streaming, feature_store, train, serving, monitoring
"""

__version__ = "0.1.0"

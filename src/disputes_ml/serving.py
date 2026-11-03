"""Real-time Model Serving endpoint for the champion model, and a small client."""

from __future__ import annotations

import math
from datetime import timezone

import pandas as pd

from .config import Config


def deploy_endpoint(cfg: Config, version: int | None = None, inference_table: bool = True):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        AiGatewayConfig,
        AiGatewayInferenceTableConfig,
        EndpointCoreConfigInput,
        ServedEntityInput,
    )
    from mlflow.tracking import MlflowClient

    if version is None:
        version = int(MlflowClient(registry_uri="databricks-uc")
                      .get_model_version_by_alias(cfg.uc_model, "champion").version)
    w = WorkspaceClient()
    served = [ServedEntityInput(
        entity_name=cfg.uc_model, entity_version=str(version),
        workload_size="Small", scale_to_zero_enabled=True,
    )]
    existing = {e.name for e in w.serving_endpoints.list()}
    if cfg.endpoint_name in existing:
        w.serving_endpoints.update_config_and_wait(name=cfg.endpoint_name, served_entities=served)
    else:
        gateway = None
        if inference_table:  # request/response log as a UC Delta table
            gateway = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(
                enabled=True, catalog_name=cfg.catalog, schema_name=cfg.schema,
                table_name_prefix=cfg.model_name,
            ))
        w.serving_endpoints.create_and_wait(
            name=cfg.endpoint_name,
            config=EndpointCoreConfigInput(name=cfg.endpoint_name, served_entities=served),
            ai_gateway=gateway,
        )
    return version


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "isoformat"):
    # The model signature has naive timestamps: send UTC without an offset.
        if getattr(v, "tzinfo", None) is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v.isoformat()
    return v.item() if hasattr(v, "item") else v


def score(cfg: Config, requests: pd.DataFrame, batch_size: int = 500) -> pd.DataFrame:
    """POST rows to the endpoint. With the online store, rows only need card_uid,
    lookup_ts and the request fields; card features are looked up server-side."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    out = []
    for i in range(0, len(requests), batch_size):
        chunk = requests.iloc[i:i + batch_size]
        records = [{k: _clean(v) for k, v in r.items()} for r in chunk.to_dict("records")]
        resp = w.serving_endpoints.query(name=cfg.endpoint_name, dataframe_records=records)
        out.append(pd.DataFrame(resp.predictions, index=chunk.index))
    return pd.concat(out) if out else pd.DataFrame()

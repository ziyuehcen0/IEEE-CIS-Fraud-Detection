"""MLflow pyfunc around RiskScorer. Shipped with the model via `code_paths`."""

from __future__ import annotations

import mlflow.pyfunc

from .scorer import RiskScorer


class ChargebackRiskModel(mlflow.pyfunc.PythonModel):
    def __init__(self, scorer: RiskScorer, version_tag: str = ""):
        self.scorer = scorer
        self.version_tag = version_tag

    def predict(self, context, model_input, params=None):
        out = self.scorer.score(model_input)
        out["model_tag"] = self.version_tag
        return out

"""Single source of truth for names, windows and split boundaries."""

from __future__ import annotations

from dataclasses import dataclass

# IEEE-CIS TransactionDT is "seconds from a reference datetime" that Vesta did not publish.
# The community convention (and the one used here) is 2017-12-01 00:00:00 UTC.
REF_EPOCH = 1_512_086_400  # 2017-12-01T00:00:00Z

HOUR = 3_600
DAY = 86_400

# Rolling windows for card-level features: (t - W, t], i.e. state *after* the txn at t.
WINDOWS = {"1h": HOUR, "24h": DAY, "7d": 7 * DAY}
MAX_LOOKBACK = max(WINDOWS.values())

# Out-of-time split on TransactionDT day (IEEE-CIS train covers ~182 days).
TRAIN_END_DAY = 120   # days [0, 120)   -> training
VALID_END_DAY = 150   # days [120, 150) -> validation + threshold choice
                      # days [150, end) -> held back as the "live" stream for replay / drift


@dataclass(frozen=True)
class Config:
    catalog: str = "main"
    schema: str = "disputes_ml"
    volume: str = "raw"
    online_store: str = "disputes-online-store"
    model_name: str = "chargeback_risk"
    endpoint_name: str = "chargeback-risk"
    # Analyst capacity used to choose the alert threshold (alerts per day).
    daily_alert_capacity: int = 400
    # Days after a transaction before its chargeback label counts as mature.
    label_maturity_days: int = 60

    def t(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def landing_path(self) -> str:
        return f"{self.volume_path}/landing"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.volume_path}/_checkpoints/card_features"

    @property
    def uc_model(self) -> str:
        return self.t(self.model_name)

    # Table names
    @property
    def transactions(self) -> str:
        return self.t("transactions")          # silver, no label

    @property
    def labels(self) -> str:
        return self.t("chargeback_labels")     # label + when it became known

    @property
    def live_source(self) -> str:
        return self.t("live_source")           # held-back days for replay

    @property
    def card_features(self) -> str:
        return self.t("card_features")         # offline, time-series feature table

    @property
    def card_features_online(self) -> str:
        return self.t("card_features_online")

    @property
    def scoring_log(self) -> str:
        return self.t("scoring_log")

    @property
    def drift_baseline(self) -> str:
        return self.t("drift_baseline")

    @property
    def drift_metrics(self) -> str:
        return self.t("drift_metrics")

    @property
    def performance_metrics(self) -> str:
        return self.t("performance_metrics")

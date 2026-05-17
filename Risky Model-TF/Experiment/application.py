"""Re-export the canonical application objects under the legacy ``Experiment`` namespace."""

from risky_debt.app.application import RunAllApplication, RunAllConfig

__all__ = ["RunAllApplication", "RunAllConfig"]

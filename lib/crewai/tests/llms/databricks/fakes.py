"""Shared fakes for the Databricks provider tests."""

from __future__ import annotations

HOST = "https://dbc-test.cloud.databricks.com"


class FakeConfig:
    """Stands in for ``databricks.sdk.core.Config``: a new token per authenticate()."""

    def __init__(self, host: str = HOST) -> None:
        self.host = host
        self.auth_type = "oauth-m2m"
        self.profile = None
        self.calls = 0

    def authenticate(self) -> dict[str, str]:
        self.calls += 1
        return {"Authorization": f"Bearer tok-{self.calls}"}

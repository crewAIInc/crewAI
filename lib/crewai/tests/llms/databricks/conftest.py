from __future__ import annotations

from typing import Any

import pytest

from crewai.llms.providers.databricks import gateway

@pytest.fixture(autouse=True)
def _isolate_databricks(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for name in (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_AUTH_TYPE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    # Never read a developer's real ~/.databrickscfg.
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(tmp_path / "missing.cfg"))
    # Vendor keys must never be used for Databricks.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-be-used")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-must-not-be-used")
    gateway.reset_state()

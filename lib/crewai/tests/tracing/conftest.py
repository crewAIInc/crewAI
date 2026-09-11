import pytest


@pytest.fixture(autouse=True)
def legacy_trace_batches(monkeypatch):
    # Exercise the retained batch API independently of automatic OTel sessions.
    # Individual tests override OTEL_SDK_DISABLED, so isolate the entry point
    # rather than environment variables. Direct export has its own suite.
    monkeypatch.setattr(
        "crewai.execution._start_tracing", lambda execution_uuid, tracing: None
    )

import os


def test_get_tracer_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("CREWAI_OTEL_ENABLED", raising=False)
    from crewai.utilities.telemetry import get_tracer

    tracer = get_tracer()
    # Should not raise when creating a span even without opentelemetry installed
    with tracer.start_as_current_span("unit-test-noop"):
        pass


def test_get_tracer_no_crash_when_enabled_without_otel(monkeypatch):
    # Even if enabled, absence of OTel must not crash
    monkeypatch.setenv("CREWAI_OTEL_ENABLED", "1")
    from crewai.utilities.telemetry import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("unit-test-enabled-no-otel"):
        pass

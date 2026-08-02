import os
import threading


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


def test_ensure_provider_thread_safe_under_concurrent_init(monkeypatch):
    """Concurrent get_tracer() calls (e.g. concurrent agent instantiation)
    must not race past the readiness check and each build their own
    TracerProvider/exporter."""
    import crewai.utilities.telemetry as telemetry

    monkeypatch.setenv("CREWAI_OTEL_ENABLED", "1")
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setattr(telemetry, "_PROVIDER_READY", False)

    init_calls = []
    real_provider_cls = telemetry.TracerProvider

    class CountingTracerProvider(real_provider_cls):
        def __init__(self, *args, **kwargs):
            init_calls.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(telemetry, "TracerProvider", CountingTracerProvider)

    errors = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        try:
            tracer = telemetry.get_tracer()
            with tracer.start_as_current_span("concurrent-init-test"):
                pass
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"get_tracer() raised under concurrency: {errors}"
    assert len(init_calls) == 1, (
        f"expected exactly one TracerProvider construction, got {len(init_calls)}"
    )
    assert telemetry._PROVIDER_READY is True


def test_otlp_exporter_failure_degrades_gracefully(monkeypatch):
    """If the OTLP exporter fails to initialize (e.g. unreachable collector,
    malformed endpoint), telemetry setup must log a warning and continue
    without crashing the caller."""
    import crewai.utilities.telemetry as telemetry

    monkeypatch.setenv("CREWAI_OTEL_ENABLED", "1")
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setattr(telemetry, "_PROVIDER_READY", False)

    class ExplodingOTLPSpanExporter:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("simulated OTLP collector unreachable")

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", ExplodingOTLPSpanExporter)

    # Must not raise, even though exporter construction fails internally.
    tracer = telemetry.get_tracer()
    with tracer.start_as_current_span("otlp-failure-test"):
        pass

    assert telemetry._PROVIDER_READY is True

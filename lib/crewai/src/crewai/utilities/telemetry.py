from __future__ import annotations

import os
from typing import Any

# Optional OpenTelemetry imports (lazy / optional dependency)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    try:
        # Prefer OTLP HTTP exporter by default; works with most backends
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except Exception:  # pragma: no cover
        OTLPSpanExporter = None  # type: ignore
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - opentelemetry not installed
    trace = None  # type: ignore
    Resource = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore
    _OTEL_AVAILABLE = False


_PROVIDER_READY = False


def _enabled() -> bool:
    return os.getenv("CREWAI_OTEL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_provider() -> None:
    global _PROVIDER_READY
    if _PROVIDER_READY or not _OTEL_AVAILABLE or not _enabled():
        return

    service_name = os.getenv("SERVICE_NAME", "crewai")
    otlp_endpoint = os.getenv("OTLP_ENDPOINT")  # e.g., http://localhost:4318/v1/traces

    resource = Resource.create({"service.name": service_name}) if Resource else None
    provider = TracerProvider(resource=resource) if TracerProvider else None

    if provider and BatchSpanProcessor and OTLPSpanExporter and otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    if provider and trace:
        trace.set_tracer_provider(provider)
        _PROVIDER_READY = True


class _NoopSpanCtx:
    def __enter__(self) -> "_NoopSpanCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        return None

    def set_status(self, *_: Any, **__: Any) -> None:
        return None

    def record_exception(self, _exc: BaseException) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, _name: str):
        return _NoopSpanCtx()


_NOOP_TRACER = _NoopTracer()


def get_tracer():
    """Return an OpenTelemetry tracer if enabled and available, else a no-op tracer."""
    _ensure_provider()
    if not _OTEL_AVAILABLE or not _enabled() or not trace:
        return _NOOP_TRACER
    return trace.get_tracer("crewai")


def record_exception(span: Any, exc: BaseException) -> None:
    """Best-effort exception recording that degrades gracefully without OTel."""
    try:
        span.record_exception(exc)  # type: ignore[attr-defined]
        if Status and StatusCode:
            span.set_status(Status(StatusCode.ERROR))  # type: ignore[attr-defined]
    except Exception:
        # Swallow to keep telemetry non-fatal
        pass

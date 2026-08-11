"""crewai-core must report the same process context as crewai.

The CLI emits deployment, template and flow-creation spans through
``crewai_core.telemetry`` in processes that never import ``crewai``. Those spans
previously carried ``coding_agent``/``runtime_context`` only by accident - they
rode the global TracerProvider that ``crewai`` installed at import - so a
CLI-only process reported neither.

The detection and common-attribute behaviour itself is covered against the
shared implementation in ``lib/crewai/tests/telemetry/``; only what is specific
to crewai-core standing alone is tested here.
"""

from __future__ import annotations

from collections.abc import Iterator

from crewai_core.runtime_env import (
    CODING_AGENT_ENV_MARKERS,
    GENERIC_AGENT_ENV_VARS,
    RUNTIME_CONTEXT_ENV_MARKERS,
    detect_coding_agent,
    detect_runtime_context,
)
from crewai_core.telemetry import Telemetry, common_span_attributes
from opentelemetry.sdk.trace.export import SpanExportResult
import pytest


class _NullExporter:
    """Stands in for the OTLP exporter so no test attempts a real export."""

    def export(self, spans: object) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    common_span_attributes.cache_clear()
    yield
    common_span_attributes.cache_clear()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every marker the detectors read, so results are deterministic."""
    for _, env_vars in (*CODING_AGENT_ENV_MARKERS, *RUNTIME_CONTEXT_ENV_MARKERS):
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
    for var in (*GENERIC_AGENT_ENV_VARS, "TERM_PROGRAM", "TERMINAL_EMULATOR"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_detection_works_from_crewai_core(clean_env: pytest.MonkeyPatch) -> None:
    """The detectors resolve here, not only through the crewai re-export."""
    clean_env.setenv("CLAUDECODE", "1")
    clean_env.setenv("GITHUB_ACTIONS", "true")

    assert detect_coding_agent() == "claude_code"
    assert detect_runtime_context() == "ci"


def test_runtime_env_does_not_import_crewai() -> None:
    """crewai-core is the leaf package; importing crewai here would invert it.

    Asserted on the module's imports rather than on ``sys.modules``, which
    reflects whatever else the test session has already loaded and would make
    this depend on ordering.
    """
    import ast
    import inspect

    from crewai_core import runtime_env

    tree = ast.parse(inspect.getsource(runtime_env))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not [
        module
        for module in imported
        if module == "crewai" or module.startswith("crewai.")
    ]


def test_cli_spans_carry_the_process_context(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: CLI spans must carry coding_agent and runtime_context."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    clean_env.setenv("CLAUDECODE", "1")
    clean_env.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)

    Telemetry._instance = None
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)
    # Patched before construction: __init__ wires the real OTLP exporter.
    monkeypatch.setattr(
        "crewai_core.telemetry.SafeOTLPSpanExporter",
        lambda **_kwargs: _NullExporter(),
    )
    telemetry = Telemetry()

    exporter = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        telemetry.feature_usage_span("cli_usage:deploy")
        telemetry.template_installed_span("crew-template")
    finally:
        telemetry.provider.shutdown()
        Telemetry._instance = None

    exported = exporter.get_finished_spans()
    assert [span.name for span in exported] == ["Feature Usage", "Template Installed"]
    for span in exported:
        attributes = span.attributes
        assert attributes is not None
        assert attributes["coding_agent"] == "claude_code"
        assert attributes["runtime_context"] == "ci"

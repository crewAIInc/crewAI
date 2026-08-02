"""Tests for AI coding assistant detection in telemetry."""

import pytest

from crewai.telemetry.utils import KNOWN_CODING_AGENTS, detect_coding_agent


ALL_MARKERS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CURSOR_TRACE_ID",
    "CURSOR_AGENT",
    "CODEX_SANDBOX",
    "CODEX_SANDBOX_NETWORK_DISABLED",
    "GEMINI_CLI",
    "AIDER_MODEL",
    "WINDSURF_SESSION_ID",
    "DEVIN_SESSION_ID",
    "REPLIT_AGENT",
    "COPILOT_AGENT_ID",
    "GITHUB_COPILOT_CLI",
    "OPENHANDS_SESSION_ID",
    "CLINE_ACTIVE",
    "AMP_AGENT",
    "TERM_PROGRAM",
    "TERMINAL_EMULATOR",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every marker so each test starts from a known state."""
    for var in ALL_MARKERS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    ("env_var", "expected"),
    [
        ("CLAUDECODE", "claude_code"),
        ("CLAUDE_CODE_ENTRYPOINT", "claude_code"),
        ("CURSOR_TRACE_ID", "cursor"),
        ("CURSOR_AGENT", "cursor"),
        ("CODEX_SANDBOX", "codex"),
        ("GEMINI_CLI", "gemini_cli"),
        ("AIDER_MODEL", "aider"),
        ("WINDSURF_SESSION_ID", "windsurf"),
        ("DEVIN_SESSION_ID", "devin"),
        ("REPLIT_AGENT", "replit_agent"),
        ("COPILOT_AGENT_ID", "copilot"),
        ("OPENHANDS_SESSION_ID", "openhands"),
        ("CLINE_ACTIVE", "cline"),
        ("AMP_AGENT", "amp_code"),
    ],
)
def test_detects_each_coding_agent(clean_env, env_var, expected):
    clean_env.setenv(env_var, "1")
    assert detect_coding_agent() == expected


def test_editor_terminal_requires_exact_value(clean_env):
    clean_env.setenv("TERM_PROGRAM", "vscode")
    assert detect_coding_agent() == "vscode_terminal"

    clean_env.setenv("TERM_PROGRAM", "iTerm.app")
    assert detect_coding_agent() != "vscode_terminal"


def test_explicit_agent_marker_wins_over_editor_terminal(clean_env):
    clean_env.setenv("TERM_PROGRAM", "vscode")
    clean_env.setenv("CLAUDECODE", "1")
    assert detect_coding_agent() == "claude_code"


def test_empty_marker_value_is_ignored(clean_env):
    clean_env.setenv("CLAUDECODE", "")
    assert detect_coding_agent() != "claude_code"


def test_falls_back_to_non_interactive_without_tty(clean_env, monkeypatch):
    monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: False})())
    assert detect_coding_agent() == "non_interactive"


def test_falls_back_to_unknown_with_tty(clean_env, monkeypatch):
    monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: True})())
    assert detect_coding_agent() == "unknown"


def test_never_returns_env_var_value(clean_env):
    """The detected name must never leak the environment variable's contents."""
    secret = "sk-super-secret-token"
    clean_env.setenv("CURSOR_TRACE_ID", secret)
    assert secret not in detect_coding_agent()


def test_handles_broken_stdout(clean_env, monkeypatch):
    class BrokenStdout:
        def isatty(self):
            raise ValueError("detached")

    monkeypatch.setattr("sys.stdout", BrokenStdout())
    assert detect_coding_agent() == "unknown"


def test_result_is_always_a_known_literal(clean_env):
    """PII guarantee: the return value can only ever be a known literal.

    Every marker is set to a value that would be catastrophic to emit, and the
    result must still come from the fixed vocabulary.
    """
    sensitive = "/Users/jane.doe/secrets/api-key-sk-live-1234"

    for var in ALL_MARKERS:
        clean_env.setenv(var, sensitive)
        result = detect_coding_agent()
        assert result in KNOWN_CODING_AGENTS
        assert sensitive not in result
        clean_env.delenv(var, raising=False)


def test_known_agents_contains_no_pii_shaped_values():
    """Every possible emitted value is a short, opaque identifier."""
    for name in KNOWN_CODING_AGENTS:
        assert name.replace("_", "").isalnum(), name
        assert len(name) <= 32, name


def test_coding_agent_lands_on_every_exported_span(clean_env):
    """End-to-end: the attribute must appear as a *span attribute* on any span.

    It cannot be a Resource attribute - the ingestion pipeline preserves only
    serviceName from the resource, so anything else set there is dropped before
    it reaches storage. This test exports through a real TracerProvider and
    asserts the attribute survives on arbitrary spans.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from crewai.telemetry.telemetry import CommonAttributesSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        CommonAttributesSpanProcessor({"coding_agent": "claude_code"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("crewai.telemetry")
    for name in ("Crew Created", "Task Execution", "Tool Usage", "Feature Usage"):
        span = tracer.start_span(name)
        span.end()

    exported = exporter.get_finished_spans()
    assert len(exported) == 4
    for span in exported:
        assert span.attributes["coding_agent"] == "claude_code", span.name

    # It must be a span attribute, not a resource attribute, or ingestion drops it.
    assert "coding_agent" not in exported[0].resource.attributes


def test_common_attributes_processor_never_breaks_span_creation(clean_env):
    """A failure applying attributes must not propagate into user execution."""
    from crewai.telemetry.telemetry import CommonAttributesSpanProcessor

    class ExplodingSpan:
        def set_attributes(self, _):
            raise RuntimeError("boom")

    CommonAttributesSpanProcessor({"coding_agent": "cursor"}).on_start(
        ExplodingSpan()  # type: ignore[arg-type]
    )


def test_coding_agent_span_emits_once(clean_env, monkeypatch):
    from crewai.telemetry.telemetry import Telemetry

    clean_env.setenv("CLAUDECODE", "1")

    telemetry = Telemetry()
    telemetry._coding_agent_reported = False

    emitted: list[str] = []
    monkeypatch.setattr(telemetry, "feature_usage_span", emitted.append)

    telemetry.coding_agent_span()
    telemetry.coding_agent_span()
    telemetry.coding_agent_span()

    assert emitted == ["coding_agent:claude_code"]

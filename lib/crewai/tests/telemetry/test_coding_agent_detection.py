"""Tests for AI coding assistant detection in telemetry."""

import os
from unittest.mock import patch

import pytest

from crewai.telemetry.utils import KNOWN_CODING_AGENTS, detect_coding_agent
from crewai.utilities.constants import (
    CC_ENV_VAR,
    CODEX_ENV_VARS,
    CODING_AGENT_ENV_MARKERS,
    CURSOR_ENV_VARS,
)


# Derived from the shared table rather than restated, so adding an assistant
# there cannot leave these tests silently checking a stale marker set.
ALL_MARKERS = tuple(
    var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars
) + ("TERM_PROGRAM", "TERMINAL_EMULATOR")

EVERY_MARKER_CASE = [
    (var, agent) for agent, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every marker so each test starts from a known state."""
    for var in ALL_MARKERS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def isolated_telemetry(monkeypatch):
    """Build a fresh Telemetry without touching the process-wide singleton.

    Telemetry is a singleton whose __init__ registers atexit and signal
    handlers. Re-initializing the shared instance would leak state into later
    tests and stack duplicate handlers, so replace _instance for the duration
    of the test and suppress lifecycle registration.
    """
    from crewai.telemetry.telemetry import Telemetry

    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)

    def build():
        with patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ):
            return Telemetry()

    yield build

    Telemetry._instance = None


@pytest.mark.parametrize(("env_var", "expected"), EVERY_MARKER_CASE)
def test_detects_every_marker_in_the_shared_table(clean_env, env_var, expected):
    """Every marker must map to its assistant, including Codex/Cursor extras."""
    clean_env.setenv(env_var, "1")
    assert detect_coding_agent() == expected


def test_shares_the_canonical_marker_sets():
    """Detection must not maintain a second, narrower set of markers.

    The env-context events and telemetry previously disagreed: a session
    exposing only CODEX_THREAD_ID was Codex to get_env_context() but unknown
    here. Both now read the same table.
    """
    by_agent = dict(CODING_AGENT_ENV_MARKERS)

    assert CC_ENV_VAR in by_agent["claude_code"]
    assert by_agent["codex"] is CODEX_ENV_VARS
    assert by_agent["cursor"] is CURSOR_ENV_VARS


def test_codex_takes_precedence_over_cursor(clean_env):
    """Codex running inside Cursor must report codex, matching get_env_context().

    Cursor sets CURSOR_* in every integrated terminal, so checking Cursor first
    would mask any assistant spawned inside it.
    """
    clean_env.setenv("CURSOR_TRACE_ID", "t-1")
    clean_env.setenv("CODEX_THREAD_ID", "th-1")

    assert detect_coding_agent() == "codex"


def test_claude_code_takes_precedence_over_cursor(clean_env):
    clean_env.setenv("CURSOR_TRACE_ID", "t-1")
    clean_env.setenv("CLAUDECODE", "1")

    assert detect_coding_agent() == "claude_code"


def test_precedence_matches_get_env_context(clean_env):
    """The two signals must agree on which assistant is present."""
    from crewai.events.types.env_events import (
        CCEnvEvent,
        CodexEnvEvent,
        CursorEnvEvent,
    )
    from crewai.utilities import env as env_module

    event_to_agent = {
        CCEnvEvent: "claude_code",
        CodexEnvEvent: "codex",
        CursorEnvEvent: "cursor",
    }

    for markers in (
        {"CLAUDECODE": "1"},
        {"CODEX_THREAD_ID": "1"},
        {"CURSOR_TRACE_ID": "1"},
        {"CURSOR_TRACE_ID": "1", "CODEX_CI": "1"},
        {"CURSOR_SANDBOX": "1", "CLAUDECODE": "1"},
    ):
        for var in ALL_MARKERS:
            clean_env.delenv(var, raising=False)
        for var, value in markers.items():
            clean_env.setenv(var, value)

        emitted: list[type] = []
        clean_env.setattr(
            env_module.crewai_event_bus,
            "emit",
            lambda _source, event, sink=emitted: sink.append(type(event)),
        )
        env_module._env_context_emitted.set(False)
        env_module.get_env_context()

        expected = event_to_agent[emitted[0]]
        assert detect_coding_agent() == expected, markers


def test_config_style_variables_are_not_used_as_markers():
    """Persistent user config must never be treated as a session marker.

    crewai loads dotenv files on normal runs, so a committed AIDER_MODEL or
    similar would mislabel ordinary human executions.
    """
    all_vars = {var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars}

    assert "AIDER_MODEL" not in all_vars


def test_every_marker_comes_from_a_verified_set():
    """Guard against reintroducing guessed variable names.

    A wrong name never matches, so the assistant is silently counted as
    "unknown" while the table implies it is covered - worse than omitting it.
    Adding an assistant means extending the canonical sets, which keeps both
    detection paths in sync.
    """
    verified = {CC_ENV_VAR, *CODEX_ENV_VARS, *CURSOR_ENV_VARS}
    declared = {var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars}

    assert declared == verified, (
        "markers must come from CC_ENV_VAR / CODEX_ENV_VARS / CURSOR_ENV_VARS; "
        f"unverified names present: {sorted(declared - verified)}"
    )


def test_concurrent_attach_registers_the_processor_once(isolated_telemetry, clean_env):
    """Check-then-act on the provider set must be locked.

    Crews and flows created from different threads can both reach set_tracer()
    before trace_set flips, and without a lock each would attach its own
    processor to the same provider for the life of the process.
    """
    import threading
    import time

    telemetry = isolated_telemetry()
    threads_count = 8

    class SlowProvider:
        """Widens the check-then-act window so the race is deterministic.

        Sleeping inside add_span_processor guarantees every unlocked thread gets
        past the membership check before any of them records the provider.
        """

        def __init__(self) -> None:
            self.processors: list[object] = []

        def add_span_processor(self, processor: object) -> None:
            time.sleep(0.05)
            self.processors.append(processor)

    provider = SlowProvider()
    start = threading.Barrier(threads_count)

    def attach() -> None:
        start.wait()
        telemetry._attach_common_attributes(provider)

    threads = [threading.Thread(target=attach) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(provider.processors) == 1


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


def test_coding_agent_span_emits_once(isolated_telemetry, clean_env, monkeypatch):
    clean_env.setenv("CLAUDECODE", "1")

    telemetry = isolated_telemetry()

    emitted: list[str] = []
    monkeypatch.setattr(telemetry, "feature_usage_span", emitted.append)

    telemetry.coding_agent_span()
    telemetry.coding_agent_span()
    telemetry.coding_agent_span()

    assert emitted == ["coding_agent:claude_code"]


def test_attribute_survives_an_externally_installed_provider(
    isolated_telemetry, clean_env
):
    """Spans must keep coding_agent when the app installs its own provider.

    set_tracer() leaves an existing non-proxy provider in place, and telemetry
    methods resolve their tracer through the global provider - so attaching the
    processor only to our own provider would drop the attribute entirely in any
    already-instrumented application.
    """
    from opentelemetry import trace as ot
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    clean_env.setenv("CLAUDECODE", "1")

    exporter = InMemorySpanExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(exporter))

    with patch.object(ot, "get_tracer_provider", return_value=app_provider):
        telemetry = isolated_telemetry()
        telemetry.set_tracer()

    span = app_provider.get_tracer("crewai.telemetry").start_span("Crew Created")
    span.end()

    exported = exporter.get_finished_spans()
    assert len(exported) == 1
    assert exported[0].attributes["coding_agent"] == "claude_code"


def test_attaching_common_attributes_is_idempotent(isolated_telemetry, clean_env):
    """Repeated set_tracer() calls must not stack duplicate processors."""
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    telemetry = isolated_telemetry()

    before = len(provider._active_span_processor._span_processors)
    telemetry._attach_common_attributes(provider)
    telemetry._attach_common_attributes(provider)
    after = len(provider._active_span_processor._span_processors)

    assert after - before == 1


def test_attaching_to_a_provider_without_processors_is_safe(isolated_telemetry):
    """A NoOp provider has no add_span_processor; this must not raise."""
    telemetry = isolated_telemetry()

    telemetry._attach_common_attributes(object())

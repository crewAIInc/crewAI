"""Tests for AI coding assistant detection in telemetry."""

import os
from unittest.mock import patch

import pytest

from crewai.telemetry.constants import TRACER_NAME
from crewai.telemetry.utils import (
    KNOWN_CODING_AGENTS,
    KNOWN_RUNTIME_CONTEXTS,
    detect_coding_agent,
    detect_runtime_context,
)
from crewai.utilities.constants import (
    ANTIGRAVITY_ENV_VARS,
    AUGMENT_ENV_VARS,
    CC_ENV_VAR,
    CC_ENV_VARS,
    CLINE_ENV_VARS,
    CODEX_ENV_VARS,
    CODING_AGENT_ENV_MARKERS,
    CURSOR_ENV_VARS,
    GEMINI_CLI_ENV_VARS,
    GENERIC_AGENT_ENV_VARS,
    JUNIE_ENV_VARS,
    OPENCODE_ENV_VARS,
    RUNTIME_CONTEXT_ENV_MARKERS,
)


# Derived from the shared tables rather than restated, so adding an assistant
# or runtime there cannot leave these tests silently checking a stale set.
RUNTIME_MARKERS = tuple(
    var for _, env_vars in RUNTIME_CONTEXT_ENV_MARKERS for var in env_vars
)

ALL_MARKERS = (
    tuple(var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars)
    + RUNTIME_MARKERS
    + GENERIC_AGENT_ENV_VARS
    + ("TERM_PROGRAM", "TERMINAL_EMULATOR")
)

EVERY_RUNTIME_CASE = [
    (var, context)
    for context, env_vars in RUNTIME_CONTEXT_ENV_MARKERS
    for var in env_vars
]

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
def otel_enabled(monkeypatch):
    """Let the SDK build real spans for tests that assert on exported ones.

    The suite runs with OTEL_SDK_DISABLED set, which makes TracerProvider hand
    out no-op tracers. An export-based assertion then sees zero spans rather
    than a failed attribute, so it fails only when it happens to run before a
    test whose fixture flips the variable - which random ordering decides.
    """
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")


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


@pytest.mark.parametrize(
    "config_var",
    [
        "AIDER_MODEL",
        "COPILOT_GITHUB_TOKEN",
        "COPILOT_MODEL",
        "GOOSE_PROVIDER",
    ],
)
def test_config_style_variables_are_not_used_as_markers(config_var):
    """Persistent user config must never be treated as a session marker.

    crewai loads dotenv files on normal runs, so a committed AIDER_MODEL or
    GOOSE_PROVIDER would mislabel ordinary human executions. Published
    detection matrices list several of these; they are deliberately excluded
    here rather than copied wholesale.
    """
    all_vars = {var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars}

    assert config_var not in all_vars


def test_hosted_environments_are_not_reported_as_assistants():
    """REPL_ID marks a hosted environment, not an assistant driving the run."""
    all_vars = {var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars}

    assert "REPL_ID" not in all_vars


def test_every_marker_comes_from_a_verified_set():
    """Guard against reintroducing guessed variable names.

    A wrong name never matches, so the assistant is silently counted as
    "unknown" while the table implies it is covered - worse than omitting it.
    Adding an assistant means extending the canonical sets, which keeps both
    detection paths in sync.
    """
    verified = {
        *ANTIGRAVITY_ENV_VARS,
        *AUGMENT_ENV_VARS,
        *CC_ENV_VARS,
        *CLINE_ENV_VARS,
        *CODEX_ENV_VARS,
        *CURSOR_ENV_VARS,
        *GEMINI_CLI_ENV_VARS,
        *JUNIE_ENV_VARS,
        *OPENCODE_ENV_VARS,
    }
    declared = {var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars}

    assert declared == verified, (
        "markers must come from the canonical per-assistant sets; "
        f"unverified names present: {sorted(declared - verified)}"
    )


def test_generic_marker_is_the_last_resort(clean_env):
    """AI_AGENT says an assistant is present without naming which one.

    A named marker must win, so the generic entry cannot mask a specific one.
    """
    clean_env.setenv("AI_AGENT", "1")
    assert detect_coding_agent() == "other"

    clean_env.setenv("CLINE_ACTIVE", "true")
    assert detect_coding_agent() == "cline"


def test_generic_marker_value_is_never_reported(clean_env):
    """Its value is an arbitrary vendor string, so it is never read."""
    clean_env.setenv("AI_AGENT", "some-unreleased-tool/2.0")

    assert detect_coding_agent() == "other"


def test_terminal_bound_assistants_outrank_cursor(clean_env):
    """CURSOR_* is set for every integrated terminal.

    Checking Cursor first would report cursor for anything spawned inside it,
    the same trap Codex already had to be ordered around.
    """
    clean_env.setenv("CURSOR_TRACE_ID", "t-1")

    for marker, expected in (
        ("CLINE_ACTIVE", "cline"),
        ("GEMINI_CLI", "gemini_cli"),
        ("AUGMENT_AGENT", "augment"),
        ("OPENCODE_CLIENT", "opencode"),
    ):
        clean_env.setenv(marker, "1")
        assert detect_coding_agent() == expected, marker
        clean_env.delenv(marker)


def test_editor_terminal_requires_exact_value(clean_env):
    clean_env.setenv("TERM_PROGRAM", "vscode")
    assert detect_runtime_context() == "vscode_terminal"

    clean_env.setenv("TERM_PROGRAM", "iTerm.app")
    assert detect_runtime_context() != "vscode_terminal"


def test_editor_terminal_is_not_reported_as_an_assistant(clean_env):
    """An editor's terminal says where a process runs, not who drove it."""
    clean_env.setenv("TERM_PROGRAM", "vscode")
    assert detect_coding_agent() == "unknown"


def test_empty_marker_value_is_ignored(clean_env):
    clean_env.setenv("CLAUDECODE", "")
    assert detect_coding_agent() != "claude_code"


@pytest.mark.parametrize(("env_var", "expected"), EVERY_RUNTIME_CASE)
def test_detects_every_runtime_marker(clean_env, env_var, expected):
    """Every runtime marker must map to its context."""
    clean_env.setenv(env_var, "1")
    assert detect_runtime_context() == expected


def test_runtime_precedence_prefers_the_most_specific(clean_env):
    """CI and hosted IDEs usually run in containers; the specific one wins."""
    clean_env.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert detect_runtime_context() == "container"

    clean_env.setenv("GITHUB_ACTIONS", "true")
    assert detect_runtime_context() == "ci"


def test_an_automated_run_still_reports_an_unknown_assistant(clean_env):
    """The split must keep the two fields independent.

    A CI run has no assistant to find, which is different from failing to
    recognize one - the reason they no longer share a field.
    """
    clean_env.setenv("CI", "true")
    assert detect_runtime_context() == "ci"
    assert detect_coding_agent() == "unknown"


def test_assistant_and_runtime_are_reported_together(clean_env):
    """An assistant inside CI must not mask either signal."""
    clean_env.setenv("CI", "true")
    clean_env.setenv("CLAUDECODE", "1")
    assert detect_coding_agent() == "claude_code"
    assert detect_runtime_context() == "ci"


def test_falls_back_to_non_interactive_without_tty(clean_env, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: False})())
    assert detect_runtime_context() == "non_interactive"


def test_falls_back_to_interactive_with_tty(clean_env, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("sys.stdout", type("S", (), {"isatty": lambda self: True})())
    assert detect_runtime_context() == "interactive"


def test_dockerenv_marks_a_container(clean_env, monkeypatch):
    """The container check is the last resort before the TTY fallback."""
    monkeypatch.setattr("os.path.exists", lambda path: path == "/.dockerenv")
    assert detect_runtime_context() == "container"


def test_unmatched_assistant_is_unknown(clean_env, monkeypatch):
    """No marker means a gap in the table, reported as unknown."""
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

    monkeypatch.setattr("os.path.exists", lambda path: False)
    monkeypatch.setattr("sys.stdout", BrokenStdout())
    assert detect_runtime_context() == "unknown"


def test_result_is_always_a_known_literal(clean_env):
    """PII guarantee: the return value can only ever be a known literal.

    Every marker is set to a value that would be catastrophic to emit, and the
    result must still come from the fixed vocabulary.
    """
    sensitive = "/Users/jane.doe/secrets/api-key-sk-live-1234"

    for var in ALL_MARKERS:
        clean_env.setenv(var, sensitive)
        agent = detect_coding_agent()
        context = detect_runtime_context()
        assert agent in KNOWN_CODING_AGENTS
        assert context in KNOWN_RUNTIME_CONTEXTS
        assert sensitive not in agent
        assert sensitive not in context
        clean_env.delenv(var, raising=False)


def test_known_agents_contains_no_pii_shaped_values():
    """Every possible emitted value is a short, opaque identifier."""
    for name in KNOWN_CODING_AGENTS | KNOWN_RUNTIME_CONTEXTS:
        assert name.replace("_", "").isalnum(), name
        assert len(name) <= 32, name


def test_coding_agent_lands_on_every_exported_span(clean_env, otel_enabled):
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

    from crewai_core.telemetry import CommonAttributesSpanProcessor

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


def test_common_attributes_processor_never_breaks_span_creation(clean_env, otel_enabled):
    """A failure applying attributes must not propagate into user execution."""
    from crewai_core.telemetry import CommonAttributesSpanProcessor

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


def test_common_attributes_land_on_our_own_spans(
    isolated_telemetry, clean_env, otel_enabled
):
    """Spans built from our own provider must carry the process-wide context."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    clean_env.setenv("CLAUDECODE", "1")

    telemetry = isolated_telemetry()
    exporter = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    telemetry.provider.get_tracer(TRACER_NAME).start_span("Crew Created").end()

    exported = exporter.get_finished_spans()
    assert len(exported) == 1
    assert exported[0].attributes["coding_agent"] == "claude_code"


def test_an_app_provider_never_receives_our_processor(
    isolated_telemetry, clean_env, otel_enabled
):
    """An application's own provider must be left completely alone.

    Attaching our processor to it stamped CrewAI's process-wide attributes onto
    that application's unrelated spans - HTTP handlers, DB queries - which is
    both wrong data and data we were never given permission to annotate.
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

    app_provider.get_tracer("some.app.library").start_span("GET /status").end()

    exported = exporter.get_finished_spans()
    assert len(exported) == 1
    assert "coding_agent" not in (exported[0].attributes or {})
    assert "project_id" not in (exported[0].attributes or {})


def _common_attributes(monkeypatch, project_id=None):
    """Build the process-wide span attributes with a stubbed project id."""
    from crewai_core.telemetry import common_span_attributes

    monkeypatch.setattr(
        "crewai_core.telemetry.get_project_id", lambda *a, **k: project_id
    )
    common_span_attributes.cache_clear()
    return common_span_attributes()


def test_common_attributes_carry_agent_and_runtime(clean_env, monkeypatch):
    """Both fields ride on every span, independently of each other."""
    clean_env.setenv("CLAUDECODE", "1")
    clean_env.setenv("GITHUB_ACTIONS", "true")

    attributes = _common_attributes(monkeypatch)

    assert attributes["coding_agent"] == "claude_code"
    assert attributes["runtime_context"] == "ci"


def test_common_attributes_include_project_id_when_declared(clean_env, monkeypatch):
    attributes = _common_attributes(monkeypatch, project_id="proj-123")

    assert attributes["project_id"] == "proj-123"


def test_project_id_is_omitted_when_absent(clean_env, monkeypatch):
    """Projects without an id must not report a placeholder."""
    attributes = _common_attributes(monkeypatch, project_id=None)

    assert "project_id" not in attributes


def test_project_id_lookup_never_breaks_telemetry(clean_env, monkeypatch):
    """A failed lookup degrades to omitting the attribute."""
    from crewai_core.telemetry import common_span_attributes

    def boom(*args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr("crewai_core.telemetry.get_project_id", boom)
    common_span_attributes.cache_clear()

    attributes = common_span_attributes()

    assert "project_id" not in attributes
    assert "coding_agent" in attributes


def test_common_attributes_are_computed_once(clean_env, monkeypatch):
    """The project file must not be re-read for each provider."""
    from crewai_core.telemetry import common_span_attributes

    calls = []

    def counting_get_project_id(*args, **kwargs):
        calls.append(1)
        return "proj-123"

    monkeypatch.setattr(
        "crewai_core.telemetry.get_project_id", counting_get_project_id
    )
    common_span_attributes.cache_clear()

    first = common_span_attributes()
    second = common_span_attributes()

    assert first is second
    assert len(calls) == 1


def test_all_common_attributes_land_on_exported_spans(clean_env, monkeypatch, otel_enabled):
    """End-to-end: every common attribute survives onto arbitrary spans."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from crewai_core.telemetry import CommonAttributesSpanProcessor

    clean_env.setenv("CLAUDECODE", "1")
    clean_env.setenv("CI", "true")
    attributes = _common_attributes(monkeypatch, project_id="proj-123")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(CommonAttributesSpanProcessor(attributes))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    provider.get_tracer("test").start_span("Feature Usage").end()
    provider.force_flush()

    exported = dict(exporter.get_finished_spans()[0].attributes)
    assert exported["coding_agent"] == "claude_code"
    assert exported["runtime_context"] == "ci"
    assert exported["project_id"] == "proj-123"


def test_runtime_markers_are_detected_by_presence(clean_env, monkeypatch):
    """An empty value still means the platform set the marker.

    Some platforms export a bare `CI=`; truthiness checks would drop those
    runs to the TTY fallback and mislabel them as ordinary local executions.
    """
    monkeypatch.setattr("os.path.exists", lambda path: False)
    clean_env.setenv("CI", "")

    assert detect_runtime_context() == "ci"


def test_managed_platforms_are_not_reported_as_serverless(clean_env):
    """Long-lived managed platforms must not claim the serverless label.

    DYNO and WEBSITE_INSTANCE_ID mark Heroku dynos and Azure App Service
    instances, which are containers rather than per-invocation functions.
    """
    clean_env.setenv("DYNO", "web.1")
    assert detect_runtime_context() == "paas"

    clean_env.delenv("DYNO")
    clean_env.setenv("WEBSITE_INSTANCE_ID", "abc123")
    assert detect_runtime_context() == "paas"


def test_serverless_markers_still_win_over_paas(clean_env):
    clean_env.setenv("DYNO", "web.1")
    clean_env.setenv("AWS_LAMBDA_FUNCTION_NAME", "my-fn")

    assert detect_runtime_context() == "serverless"


def test_generic_marker_is_detected_by_presence(clean_env):
    """An empty AI_AGENT still says an assistant is present.

    The named markers keep truthiness, where an empty value means the tool set
    a placeholder rather than claiming the session.
    """
    clean_env.setenv("AI_AGENT", "")

    assert detect_coding_agent() == "other"


def test_azure_functions_are_not_reported_as_paas(clean_env):
    """Azure Functions run on the App Service host and inherit its marker."""
    clean_env.setenv("WEBSITE_INSTANCE_ID", "abc123")
    clean_env.setenv("FUNCTIONS_WORKER_RUNTIME", "python")

    assert detect_runtime_context() == "serverless"


def test_env_context_precedence_matches_the_shared_table(clean_env):
    """Both detection paths must agree on which assistant is present.

    get_env_context previously restated precedence, so a marker added for
    telemetry was invisible here and the two disagreed.
    """
    from crewai.events.types.env_events import (
        CCEnvEvent,
        CodexEnvEvent,
        CursorEnvEvent,
        DefaultEnvEvent,
    )
    from crewai.utilities import env as env_module

    agent_to_event = {
        "claude_code": CCEnvEvent,
        "codex": CodexEnvEvent,
        "cursor": CursorEnvEvent,
    }

    for agent, env_vars in CODING_AGENT_ENV_MARKERS:
        for var in env_vars:
            for other in ALL_MARKERS:
                clean_env.delenv(other, raising=False)
            clean_env.setenv(var, "1")

            emitted: list[type] = []
            clean_env.setattr(
                env_module.crewai_event_bus,
                "emit",
                lambda _source, event, sink=emitted: sink.append(type(event)),
            )
            env_module._env_context_emitted.set(False)
            env_module.get_env_context()

            assert detect_coding_agent() == agent, var
            assert emitted[0] is agent_to_event.get(agent, DefaultEnvEvent), var


def test_assistant_inside_cursor_agrees_across_both_paths(clean_env):
    """Cursor sets CURSOR_* in every terminal, including for other assistants."""
    from crewai.events.types.env_events import DefaultEnvEvent
    from crewai.utilities import env as env_module

    clean_env.setenv("CURSOR_TRACE_ID", "t-1")
    clean_env.setenv("CLINE_ACTIVE", "true")

    emitted: list[type] = []
    clean_env.setattr(
        env_module.crewai_event_bus,
        "emit",
        lambda _source, event, sink=emitted: sink.append(type(event)),
    )
    env_module._env_context_emitted.set(False)
    env_module.get_env_context()

    assert detect_coding_agent() == "cline"
    assert emitted[0] is DefaultEnvEvent

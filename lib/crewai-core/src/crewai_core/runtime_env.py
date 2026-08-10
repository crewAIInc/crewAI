"""Detection of the coding assistant and runtime a process is running under.

Lives in ``crewai_core`` rather than ``crewai`` because both telemetry
implementations need it: the CLI emits deployment, template and flow-creation
spans through ``crewai_core.telemetry`` without importing ``crewai`` at all.

``crewai.utilities.constants`` and ``crewai.telemetry.utils`` re-export from
here, so the original import paths keep working.
"""

from __future__ import annotations

import os
import sys
from typing import Final


CC_ENV_VAR: Final[str] = "CLAUDECODE"
CODEX_ENV_VARS: Final[tuple[str, ...]] = (
    "CODEX_CI",
    "CODEX_MANAGED_BY_NPM",
    "CODEX_SANDBOX",
    "CODEX_SANDBOX_NETWORK_DISABLED",
    "CODEX_THREAD_ID",
)
CC_ENV_VARS: Final[tuple[str, ...]] = (CC_ENV_VAR, "CLAUDE_CODE")
CURSOR_ENV_VARS: Final[tuple[str, ...]] = (
    "CURSOR_AGENT",
    "CURSOR_EXTENSION_HOST_ROLE",
    "CURSOR_SANDBOX",
    "CURSOR_TRACE_ID",
    "CURSOR_WORKSPACE_LABEL",
)
ANTIGRAVITY_ENV_VARS: Final[tuple[str, ...]] = (
    "ANTIGRAVITY_AGENT",
    "ANTIGRAVITY_CLI_ALIAS",
)
AUGMENT_ENV_VARS: Final[tuple[str, ...]] = ("AUGMENT_AGENT",)
CLINE_ENV_VARS: Final[tuple[str, ...]] = ("CLINE_ACTIVE",)
GEMINI_CLI_ENV_VARS: Final[tuple[str, ...]] = ("GEMINI_CLI",)
JUNIE_ENV_VARS: Final[tuple[str, ...]] = ("JUNIE_DATA", "JUNIE_SHIM_PATH")
OPENCODE_ENV_VARS: Final[tuple[str, ...]] = ("OPENCODE", "OPENCODE_CLIENT")

# Proposed cross-vendor marker (agentsmd/agents.md#136). Checked last and
# reported as "other": it says an assistant is present without naming one, and
# reading its value to find out would put an arbitrary string in telemetry.
GENERIC_AGENT_ENV_VARS: Final[tuple[str, ...]] = ("AI_AGENT",)

# Ordered (name, env vars) pairs for identifying the AI coding assistant a
# process is running under. Reuses the sets above and keeps the same precedence
# as ``get_env_context()``, so the env-context events and telemetry never
# disagree about which assistant is present.
#
# Deliberately limited to assistants whose markers are verified. Guessing a
# variable name is worse than omitting the assistant: a wrong name never
# matches, so that assistant is silently counted as "unknown" while the table
# implies it is covered.
#
# Two rules for adding an entry:
#   1. Confirm the variable the tool actually sets - do not infer it from the
#      product name.
#   2. Use only *session*-scoped variables the assistant sets for processes it
#      spawns. Persistent user configuration (an ``AIDER_MODEL`` in a committed
#      ``.env``, say) is unusable: crewai loads dotenv files on normal runs, so
#      a leftover config value would mislabel ordinary human executions.
#
# Extend the shared sets above rather than adding a parallel tuple here, so both
# detection paths pick the new markers up together: ``get_env_context()`` walks
# this same table for its precedence and emits ``DefaultEnvEvent`` for the
# assistants that have no event class of their own.
#
# Markers below the first three were taken from the published detection matrix
# at vercel/detect-agent (agents.json), cross-checked against the proposal in
# agentsmd/agents.md#136 and microsoft/vscode#311734. Rule 2 excluded several
# entries those sources list: Goose's ``GOOSE_PROVIDER`` and Copilot's
# ``COPILOT_MODEL`` / ``COPILOT_GITHUB_TOKEN`` are user configuration, and a
# committed ``.env`` carrying one would relabel every ordinary run. Replit's
# ``REPL_ID`` is a hosted environment rather than an assistant, so it stays in
# HOSTED_IDE_ENV_VARS.
#
# The assistants that spawn inside another editor's terminal are ordered ahead
# of Cursor for the same reason Codex is: CURSOR_* is set for every integrated
# terminal, so checking Cursor first would mask anything running inside it.
CODING_AGENT_ENV_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("claude_code", CC_ENV_VARS),
    ("codex", CODEX_ENV_VARS),
    ("cline", CLINE_ENV_VARS),
    ("gemini_cli", GEMINI_CLI_ENV_VARS),
    ("augment", AUGMENT_ENV_VARS),
    ("opencode", OPENCODE_ENV_VARS),
    ("antigravity", ANTIGRAVITY_ENV_VARS),
    ("junie", JUNIE_ENV_VARS),
    ("cursor", CURSOR_ENV_VARS),
)

# Markers for *where* a process runs, kept separate from which assistant is
# driving it. The two answer different questions: a scheduled container run has
# no assistant to detect, and folding it into the assistant field made "no
# marker found" and "no assistant possible" indistinguishable.
#
# These are published platform contracts rather than per-tool observations, so
# they do not need the case-by-case verification the assistant table requires.
# Presence is all that is checked; no value is ever read.
CI_ENV_VARS: Final[tuple[str, ...]] = (
    "APPVEYOR",
    "BITBUCKET_BUILD_NUMBER",
    "BUILDKITE",
    "CI",
    "CIRCLECI",
    "DRONE",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
    "TRAVIS",
)
SERVERLESS_ENV_VARS: Final[tuple[str, ...]] = (
    "AWS_LAMBDA_FUNCTION_NAME",
    "FUNCTIONS_EXTENSION_VERSION",
    "FUNCTIONS_WORKER_RUNTIME",
    "FUNCTION_TARGET",
    "K_SERVICE",
    "VERCEL",
)
# Managed application platforms, kept apart from serverless: their markers are
# set for long-lived containers rather than per-invocation functions, and
# checking them under "serverless" would have claimed every Heroku dyno and
# Azure App Service instance before the container check could see them.
#
# Azure Functions run on the App Service host and inherit WEBSITE_INSTANCE_ID,
# so they would land here despite being serverless. The FUNCTIONS_* markers
# above are checked first to keep them out.
PAAS_ENV_VARS: Final[tuple[str, ...]] = (
    "DYNO",
    "WEBSITE_INSTANCE_ID",
)
HOSTED_IDE_ENV_VARS: Final[tuple[str, ...]] = (
    "CODESPACES",
    "GITPOD_WORKSPACE_ID",
    "REPL_ID",
)
NOTEBOOK_ENV_VARS: Final[tuple[str, ...]] = (
    "COLAB_RELEASE_TAG",
    "JPY_PARENT_PID",
)
CONTAINER_ENV_VARS: Final[tuple[str, ...]] = ("KUBERNETES_SERVICE_HOST",)

# Ordered most specific first. CI jobs and hosted IDEs usually run inside
# containers, so a bare container match is only meaningful once the others have
# been ruled out.
RUNTIME_CONTEXT_ENV_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("ci", CI_ENV_VARS),
    ("serverless", SERVERLESS_ENV_VARS),
    ("paas", PAAS_ENV_VARS),
    ("hosted_ide", HOSTED_IDE_ENV_VARS),
    ("notebook", NOTEBOOK_ENV_VARS),
    ("container", CONTAINER_ENV_VARS),
)


# Editors whose integrated terminal a process can be launched from. Matched on
# an exact value rather than presence, since these variables name the terminal.
_EDITOR_TERM_MARKERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("TERM_PROGRAM", "vscode", "vscode_terminal"),
    ("TERMINAL_EMULATOR", "JetBrains-JediTerm", "jetbrains_terminal"),
)

# Marks a process running inside a container when no more specific runtime
# marker is present.
_DOCKER_ENV_PATH: Final[str] = "/.dockerenv"

_UNKNOWN: Final[str] = "unknown"
_OTHER: Final[str] = "other"

# The complete set of values detect_coding_agent() can ever return. Every value
# is a literal from CODING_AGENT_ENV_MARKERS or this module, which is what makes
# the function structurally incapable of emitting PII: no environment value,
# path, hostname, or user-supplied string can reach the return value.
KNOWN_CODING_AGENTS: Final[frozenset[str]] = frozenset(
    [name for name, _ in CODING_AGENT_ENV_MARKERS] + [_OTHER, _UNKNOWN]
)

# The same guarantee for detect_runtime_context(): a closed set of literals.
KNOWN_RUNTIME_CONTEXTS: Final[frozenset[str]] = frozenset(
    [name for name, _ in RUNTIME_CONTEXT_ENV_MARKERS]
    + [name for _, _, name in _EDITOR_TERM_MARKERS]
    + ["interactive", "non_interactive", _UNKNOWN]
)


def detect_coding_agent() -> str:
    """Best-effort detection of the AI coding assistant running this process.

    Uses the shared ``CODING_AGENT_ENV_MARKERS`` table, so this agrees with the
    env-context events emitted by ``get_env_context()`` rather than maintaining
    a second, narrower set of markers. Precedence follows that table, which ends
    with Cursor: ``CURSOR_*`` is set for every integrated terminal, so checking
    it earlier would mask any assistant running inside one.

    Only the assistant's normalized name is returned - environment variable
    values are never read into the return value or recorded anywhere.

    Two limits worth knowing. This is heuristic: markers change as tools
    evolve, so "unknown" means "no known marker present", not "no agent". And
    some markers (the Cursor set in particular) are set by the editor for any
    integrated terminal, so a result names the environment the process is
    running *under*, not proof that an agent authored the code.

    Answers only *which assistant*. Where the process runs is
    :func:`detect_runtime_context`, so an "unknown" here is a genuine gap in
    the marker table rather than a run that never had an assistant to find.

    Returns:
        A normalized assistant name (e.g. "claude_code", "cursor", "codex"), or
        "unknown" when no marker matches. The result is always a member of
        KNOWN_CODING_AGENTS.
    """
    for agent_name, env_vars in CODING_AGENT_ENV_MARKERS:
        if any(os.environ.get(env_var) for env_var in env_vars):
            return agent_name

    # Checked last and by presence: the cross-vendor marker establishes that an
    # assistant is present without naming one, and an empty value still says so.
    if any(env_var in os.environ for env_var in GENERIC_AGENT_ENV_VARS):
        return _OTHER

    return _UNKNOWN


def detect_runtime_context() -> str:
    """Best-effort detection of where this process is running.

    Separate from :func:`detect_coding_agent` because the two answer different
    questions. Automated runs -- CI, containers, serverless -- have no
    assistant to detect, and reporting them in the assistant field made an
    unrecognized assistant indistinguishable from a run that could never have
    had one.

    Precedence runs most specific first: CI and hosted IDEs typically run
    inside containers, so a bare container match only applies once the more
    specific markers have been ruled out.

    Returns:
        One of the runtime names (e.g. "ci", "serverless", "container",
        "notebook", "hosted_ide"), an editor terminal (e.g. "vscode_terminal"),
        "interactive" or "non_interactive" when only a TTY check applies, or
        "unknown" when that check cannot be made. The result is always a member
        of KNOWN_RUNTIME_CONTEXTS.
    """
    # Presence, not truthiness: a platform that exports an empty CI= is still
    # CI, unlike the assistant markers where an empty value means the tool set
    # a placeholder rather than claiming the session.
    for context_name, env_vars in RUNTIME_CONTEXT_ENV_MARKERS:
        if any(env_var in os.environ for env_var in env_vars):
            return context_name

    for env_var, expected, context_name in _EDITOR_TERM_MARKERS:
        if os.environ.get(env_var) == expected:
            return context_name

    if os.path.exists(_DOCKER_ENV_PATH):
        return "container"

    try:
        return "interactive" if sys.stdout.isatty() else "non_interactive"
    except (AttributeError, ValueError, OSError):
        return _UNKNOWN

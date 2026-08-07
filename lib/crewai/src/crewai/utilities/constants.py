from typing import Annotated, Final

from crewai_core.constants import (
    CREWAI_TRAINED_AGENTS_FILE_ENV as CREWAI_TRAINED_AGENTS_FILE_ENV,
    KNOWLEDGE_DIRECTORY as KNOWLEDGE_DIRECTORY,
    MAX_FILE_NAME_LENGTH as MAX_FILE_NAME_LENGTH,
    TRAINED_AGENTS_DATA_FILE as TRAINED_AGENTS_DATA_FILE,
    TRAINING_DATA_FILE as TRAINING_DATA_FILE,
)
from crewai_core.printer import PrinterColor
from pydantic_core import CoreSchema


__all__ = [
    "ANTIGRAVITY_ENV_VARS",
    "AUGMENT_ENV_VARS",
    "CC_ENV_VAR",
    "CC_ENV_VARS",
    "CI_ENV_VARS",
    "CLINE_ENV_VARS",
    "CODEX_ENV_VARS",
    "CODING_AGENT_ENV_MARKERS",
    "CONTAINER_ENV_VARS",
    "CREWAI_TRAINED_AGENTS_FILE_ENV",
    "CURSOR_ENV_VARS",
    "EMITTER_COLOR",
    "GEMINI_CLI_ENV_VARS",
    "GENERIC_AGENT_ENV_VARS",
    "HOSTED_IDE_ENV_VARS",
    "JUNIE_ENV_VARS",
    "KNOWLEDGE_DIRECTORY",
    "MAX_FILE_NAME_LENGTH",
    "NOTEBOOK_ENV_VARS",
    "NOT_SPECIFIED",
    "OPENCODE_ENV_VARS",
    "PAAS_ENV_VARS",
    "RUNTIME_CONTEXT_ENV_MARKERS",
    "SERVERLESS_ENV_VARS",
    "TRAINED_AGENTS_DATA_FILE",
    "TRAINING_DATA_FILE",
]


EMITTER_COLOR: Final[PrinterColor] = "bold_blue"
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


class _NotSpecified:
    """Sentinel class to detect when no value has been explicitly provided.

    Notes:
        - TODO: Consider moving this class and NOT_SPECIFIED to types.py
          as they are more type-related constructs than business constants.
    """

    def __repr__(self) -> str:
        return "NOT_SPECIFIED"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: object
    ) -> CoreSchema:
        from pydantic_core import core_schema

        def _validate(v: object) -> _NotSpecified:
            if isinstance(v, _NotSpecified) or v == "NOT_SPECIFIED":
                return NOT_SPECIFIED
            raise ValueError(f"Expected NOT_SPECIFIED sentinel, got {type(v).__name__}")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: "NOT_SPECIFIED",
                info_arg=False,
            ),
        )


NOT_SPECIFIED: Final[
    Annotated[
        _NotSpecified,
        "Sentinel value used to detect when no value has been explicitly provided. "
        "Unlike `None`, which might be a valid value from the user, `NOT_SPECIFIED` "
        "allows us to distinguish between 'not passed at all' and 'explicitly passed None' or '[]'.",
    ]
] = _NotSpecified()

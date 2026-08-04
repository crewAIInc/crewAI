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
    "CC_ENV_VAR",
    "CODEX_ENV_VARS",
    "CODING_AGENT_ENV_MARKERS",
    "CREWAI_TRAINED_AGENTS_FILE_ENV",
    "CURSOR_ENV_VARS",
    "EMITTER_COLOR",
    "KNOWLEDGE_DIRECTORY",
    "MAX_FILE_NAME_LENGTH",
    "NOT_SPECIFIED",
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
CURSOR_ENV_VARS: Final[tuple[str, ...]] = (
    "CURSOR_AGENT",
    "CURSOR_EXTENSION_HOST_ROLE",
    "CURSOR_SANDBOX",
    "CURSOR_TRACE_ID",
    "CURSOR_WORKSPACE_LABEL",
)

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
# detection paths pick the new markers up together.
CODING_AGENT_ENV_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("claude_code", (CC_ENV_VAR,)),
    ("codex", CODEX_ENV_VARS),
    ("cursor", CURSOR_ENV_VARS),
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

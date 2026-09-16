from typing import Annotated, Final

from crewai_core.constants import (
    CREWAI_TRAINED_AGENTS_FILE_ENV as CREWAI_TRAINED_AGENTS_FILE_ENV,
    KNOWLEDGE_DIRECTORY as KNOWLEDGE_DIRECTORY,
    MAX_FILE_NAME_LENGTH as MAX_FILE_NAME_LENGTH,
    TRAINED_AGENTS_DATA_FILE as TRAINED_AGENTS_DATA_FILE,
    TRAINING_DATA_FILE as TRAINING_DATA_FILE,
)
from crewai_core.printer import PrinterColor

# Coding-assistant and runtime marker tables live in crewai_core so the CLI can
# detect the same context without importing crewai. Re-exported here because
# these names are the established import path.
from crewai_core.runtime_env import (
    ANTIGRAVITY_ENV_VARS as ANTIGRAVITY_ENV_VARS,
    AUGMENT_ENV_VARS as AUGMENT_ENV_VARS,
    CC_ENV_VAR as CC_ENV_VAR,
    CC_ENV_VARS as CC_ENV_VARS,
    CI_ENV_VARS as CI_ENV_VARS,
    CLINE_ENV_VARS as CLINE_ENV_VARS,
    CODEX_ENV_VARS as CODEX_ENV_VARS,
    CODING_AGENT_ENV_MARKERS as CODING_AGENT_ENV_MARKERS,
    CONTAINER_ENV_VARS as CONTAINER_ENV_VARS,
    CURSOR_ENV_VARS as CURSOR_ENV_VARS,
    GEMINI_CLI_ENV_VARS as GEMINI_CLI_ENV_VARS,
    GENERIC_AGENT_ENV_VARS as GENERIC_AGENT_ENV_VARS,
    HOSTED_IDE_ENV_VARS as HOSTED_IDE_ENV_VARS,
    JUNIE_ENV_VARS as JUNIE_ENV_VARS,
    NOTEBOOK_ENV_VARS as NOTEBOOK_ENV_VARS,
    OPENCODE_ENV_VARS as OPENCODE_ENV_VARS,
    PAAS_ENV_VARS as PAAS_ENV_VARS,
    RUNTIME_CONTEXT_ENV_MARKERS as RUNTIME_CONTEXT_ENV_MARKERS,
    SERVERLESS_ENV_VARS as SERVERLESS_ENV_VARS,
)
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

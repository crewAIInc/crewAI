from typing import Annotated, Final

from crewai.utilities.printer import PrinterColor

TRAINING_DATA_FILE: Final[str] = "training_data.pkl"
TRAINED_AGENTS_DATA_FILE: Final[str] = "trained_agents_data.pkl"
KNOWLEDGE_DIRECTORY: Final[str] = "knowledge"
MAX_FILE_NAME_LENGTH: Final[int] = 255
EMITTER_COLOR: Final[PrinterColor] = "bold_blue"


class _NotSpecified:
    """Sentinel class to detect when no value has been explicitly provided.

    Notes:
        - TODO: Consider moving this class and NOT_SPECIFIED to types.py
          as they are more type-related constructs than business constants.
    """

    def __repr__(self) -> str:
        return "NOT_SPECIFIED"


NOT_SPECIFIED: Final[
    Annotated[
        _NotSpecified,
        "Sentinel value used to detect when no value has been explicitly provided. "
        "Unlike `None`, which might be a valid value from the user, `NOT_SPECIFIED` "
        "allows us to distinguish between 'not passed at all' and 'explicitly passed None' or '[]'.",
    ]
] = _NotSpecified()
CREWAI_BASE_URL: Final[str] = "https://app.crewai.com"

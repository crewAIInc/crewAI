"""Constants shared by both crewai and crewai-cli."""

from __future__ import annotations

from typing import Final


CREWAI_TRAINED_AGENTS_FILE_ENV: Final[str] = "CREWAI_TRAINED_AGENTS_FILE"
TRAINING_DATA_FILE: Final[str] = "training_data.pkl"
TRAINED_AGENTS_DATA_FILE: Final[str] = "trained_agents_data.pkl"
KNOWLEDGE_DIRECTORY: Final[str] = "knowledge"
MAX_FILE_NAME_LENGTH: Final[int] = 255

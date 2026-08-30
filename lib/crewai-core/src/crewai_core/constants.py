"""Constants shared by both crewai and crewai-cli."""

from __future__ import annotations

from typing import Final


CREWAI_TRAINED_AGENTS_FILE_ENV: Final[str] = "CREWAI_TRAINED_AGENTS_FILE"
TRAINING_DATA_FILE: Final[str] = "training_data.pkl"
TRAINED_AGENTS_DATA_FILE: Final[str] = "trained_agents_data.pkl"
KNOWLEDGE_DIRECTORY: Final[str] = "knowledge"
MAX_FILE_NAME_LENGTH: Final[int] = 255

DEFAULT_CREWAI_ENTERPRISE_URL: Final[str] = "https://app.crewai.com"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER: Final[str] = "workos"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE: Final[str] = (
    "client_01JNJQWBJ4SPFN3SWJM5T7BDG8"
)
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID: Final[str] = (
    "client_01JYT06R59SP0NXYGD994NFXXX"
)
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN: Final[str] = "login.crewai.com"

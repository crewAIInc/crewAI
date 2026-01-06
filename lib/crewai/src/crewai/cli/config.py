import json
from logging import getLogger
from pathlib import Path
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from crewai.cli.constants import (
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER,
    DEFAULT_CREWAI_ENTERPRISE_URL,
)
from crewai.cli.shared.token_manager import TokenManager


logger = getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "crewai" / "settings.json"


def get_writable_config_path() -> Path | None:
    """
    Find a writable location for the config file with fallback options.

    Tries in order:
    1. Default: ~/.config/crewai/settings.json
    2. Temp directory: /tmp/crewai_settings.json (or OS equivalent)
    3. Current directory: ./crewai_settings.json
    4. In-memory only (returns None)

    Returns:
        Path object for writable config location, or None if no writable location found
    """
    fallback_paths = [
        DEFAULT_CONFIG_PATH,  # Default location
        Path(tempfile.gettempdir()) / "crewai_settings.json",  # Temporary directory
        Path.cwd() / "crewai_settings.json",  # Current working directory
    ]

    for config_path in fallback_paths:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            test_file = config_path.parent / ".crewai_write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Clean up test file
                logger.info(f"Using config path: {config_path}")
                return config_path
            except Exception:  # noqa: S112
                continue

        except Exception:  # noqa: S112
            continue

    return None


# Settings that are related to the user's account
USER_SETTINGS_KEYS = [
    "tool_repository_username",
    "tool_repository_password",
    "org_name",
    "org_uuid",
]

# Settings that are related to the CLI
CLI_SETTINGS_KEYS = [
    "enterprise_base_url",
    "oauth2_provider",
    "oauth2_audience",
    "oauth2_client_id",
    "oauth2_domain",
    "oauth2_extra",
]

# Default values for CLI settings
DEFAULT_CLI_SETTINGS = {
    "enterprise_base_url": DEFAULT_CREWAI_ENTERPRISE_URL,
    "oauth2_provider": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER,
    "oauth2_audience": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
    "oauth2_client_id": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    "oauth2_domain": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
    "oauth2_extra": {},
}

# Readonly settings - cannot be set by the user
READONLY_SETTINGS_KEYS = [
    "org_name",
    "org_uuid",
]

# Hidden settings - not displayed by the 'list' command and cannot be set by the user
HIDDEN_SETTINGS_KEYS = [
    "config_path",
    "tool_repository_username",
    "tool_repository_password",
]


class Settings(BaseModel):
    enterprise_base_url: str | None = Field(
        default=DEFAULT_CLI_SETTINGS["enterprise_base_url"],
        description="Base URL of the CrewAI AMP instance",
    )
    tool_repository_username: str | None = Field(
        None, description="Username for interacting with the Tool Repository"
    )
    tool_repository_password: str | None = Field(
        None, description="Password for interacting with the Tool Repository"
    )
    org_name: str | None = Field(
        None, description="Name of the currently active organization"
    )
    org_uuid: str | None = Field(
        None, description="UUID of the currently active organization"
    )
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH, frozen=True, exclude=True)

    oauth2_provider: str = Field(
        description="OAuth2 provider used for authentication (e.g., workos, okta, auth0).",
        default=DEFAULT_CLI_SETTINGS["oauth2_provider"],
    )

    oauth2_audience: str | None = Field(
        description="OAuth2 audience value, typically used to identify the target API or resource.",
        default=DEFAULT_CLI_SETTINGS["oauth2_audience"],
    )

    oauth2_client_id: str = Field(
        default=DEFAULT_CLI_SETTINGS["oauth2_client_id"],
        description="OAuth2 client ID issued by the provider, used during authentication requests.",
    )

    oauth2_domain: str = Field(
        description="OAuth2 provider's domain (e.g., your-org.auth0.com) used for issuing tokens.",
        default=DEFAULT_CLI_SETTINGS["oauth2_domain"],
    )

    oauth2_extra: dict[str, Any] = Field(
        description="Extra configuration for the OAuth2 provider.",
        default={},
    )

    def __init__(self, config_path: Path | None = None, **data: dict[str, Any]) -> None:
        """Load Settings from config path with fallback support"""
        if config_path is None:
            config_path = get_writable_config_path()

        # If config_path is None, we're in memory-only mode
        if config_path is None:
            merged_data = {**data}
            # Dummy path for memory-only mode
            super().__init__(config_path=Path("/dev/null"), **merged_data)
            return

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            merged_data = {**data}
            # Dummy path for memory-only mode
            super().__init__(config_path=Path("/dev/null"), **merged_data)
            return

        file_data = {}
        if config_path.is_file():
            try:
                with config_path.open("r") as f:
                    file_data = json.load(f)
            except Exception:
                file_data = {}

        merged_data = {**file_data, **data}
        super().__init__(config_path=config_path, **merged_data)

    def clear_user_settings(self) -> None:
        """Clear all user settings"""
        self._reset_user_settings()
        self.dump()

    def reset(self) -> None:
        """Reset all settings to default values"""
        self._reset_user_settings()
        self._reset_cli_settings()
        self._clear_auth_tokens()
        self.dump()

    def dump(self) -> None:
        """Save current settings to settings.json"""
        if str(self.config_path) == "/dev/null":
            return

        try:
            if self.config_path.is_file():
                with self.config_path.open("r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}

            updated_data = {**existing_data, **self.model_dump(exclude_unset=True)}
            with self.config_path.open("w") as f:
                json.dump(updated_data, f, indent=4)

        except Exception:  # noqa: S110
            pass

    def _reset_user_settings(self) -> None:
        """Reset all user settings to default values"""
        for key in USER_SETTINGS_KEYS:
            setattr(self, key, None)

    def _reset_cli_settings(self) -> None:
        """Reset all CLI settings to default values"""
        for key in CLI_SETTINGS_KEYS:
            setattr(self, key, DEFAULT_CLI_SETTINGS.get(key))

    def _clear_auth_tokens(self) -> None:
        """Clear all authentication tokens"""
        TokenManager().clear_tokens()

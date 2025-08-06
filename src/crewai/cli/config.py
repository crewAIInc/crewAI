import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from crewai.cli.constants import (
    DEFAULT_CREWAI_ENTERPRISE_URL,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "crewai" / "settings.json"

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
]

# Default values for CLI settings
DEFAULT_CLI_SETTINGS = {
    "enterprise_base_url": DEFAULT_CREWAI_ENTERPRISE_URL,
    "oauth2_provider": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER,
    "oauth2_audience": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
    "oauth2_client_id": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    "oauth2_domain": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
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
    enterprise_base_url: Optional[str] = Field(
        default=DEFAULT_CLI_SETTINGS["enterprise_base_url"],
        description="Base URL of the CrewAI Enterprise instance",
    )
    tool_repository_username: Optional[str] = Field(
        None, description="Username for interacting with the Tool Repository"
    )
    tool_repository_password: Optional[str] = Field(
        None, description="Password for interacting with the Tool Repository"
    )
    org_name: Optional[str] = Field(
        None, description="Name of the currently active organization"
    )
    org_uuid: Optional[str] = Field(
        None, description="UUID of the currently active organization"
    )
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH, frozen=True, exclude=True)

    oauth2_provider: str = Field(
        description="OAuth2 provider used for authentication (e.g., workos, okta, auth0).",
        default=DEFAULT_CLI_SETTINGS["oauth2_provider"]
    )

    oauth2_audience: Optional[str] = Field(
        description="OAuth2 audience value, typically used to identify the target API or resource.",
        default=DEFAULT_CLI_SETTINGS["oauth2_audience"]
    )

    oauth2_client_id: str = Field(
        default=DEFAULT_CLI_SETTINGS["oauth2_client_id"],
        description="OAuth2 client ID issued by the provider, used during authentication requests.",
    )

    oauth2_domain: str = Field(
        description="OAuth2 provider's domain (e.g., your-org.auth0.com) used for issuing tokens.",
        default=DEFAULT_CLI_SETTINGS["oauth2_domain"]
    )

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH, **data):
        """Load Settings from config path"""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        file_data = {}
        if config_path.is_file():
            try:
                with config_path.open("r") as f:
                    file_data = json.load(f)
            except json.JSONDecodeError:
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
        self.dump()

    def dump(self) -> None:
        """Save current settings to settings.json"""
        if self.config_path.is_file():
            with self.config_path.open("r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        updated_data = {**existing_data, **self.model_dump(exclude_unset=True)}
        with self.config_path.open("w") as f:
            json.dump(updated_data, f, indent=4)

    def _reset_user_settings(self) -> None:
        """Reset all user settings to default values"""
        for key in USER_SETTINGS_KEYS:
            setattr(self, key, None)

    def _reset_cli_settings(self) -> None:
        """Reset all CLI settings to default values"""
        for key in CLI_SETTINGS_KEYS:
            setattr(self, key, DEFAULT_CLI_SETTINGS.get(key))

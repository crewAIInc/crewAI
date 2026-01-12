from typing import Any, cast

import requests
from requests.exceptions import JSONDecodeError, RequestException
from rich.console import Console

from crewai.cli.authentication.main import Oauth2Settings, ProviderFactory
from crewai.cli.command import BaseCommand
from crewai.cli.settings.main import SettingsCommand
from crewai.cli.version import get_crewai_version


console = Console()


class EnterpriseConfigureCommand(BaseCommand):
    def __init__(self) -> None:
        super().__init__()
        self.settings_command = SettingsCommand()

    def configure(self, enterprise_url: str) -> None:
        try:
            enterprise_url = enterprise_url.rstrip("/")

            oauth_config = self._fetch_oauth_config(enterprise_url)

            self._update_oauth_settings(enterprise_url, oauth_config)

            console.print(
                f"✅ Successfully configured CrewAI AMP with OAuth2 settings from {enterprise_url}",
                style="bold green",
            )

        except Exception as e:
            console.print(
                f"❌ Failed to configure Enterprise settings: {e!s}", style="bold red"
            )
            raise SystemExit(1) from e

    def _fetch_oauth_config(self, enterprise_url: str) -> dict[str, Any]:
        oauth_endpoint = f"{enterprise_url}/auth/parameters"

        try:
            console.print(f"🔄 Fetching OAuth2 configuration from {oauth_endpoint}...")
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"CrewAI-CLI/{get_crewai_version()}",
                "X-Crewai-Version": get_crewai_version(),
            }
            response = requests.get(oauth_endpoint, timeout=30, headers=headers)
            response.raise_for_status()

            try:
                oauth_config = response.json()
            except JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from {oauth_endpoint}") from e

            self._validate_oauth_config(oauth_config)

            console.print(
                "✅ Successfully retrieved OAuth2 configuration", style="green"
            )
            return cast(dict[str, Any], oauth_config)

        except RequestException as e:
            raise ValueError(f"Failed to connect to enterprise URL: {e!s}") from e
        except Exception as e:
            raise ValueError(f"Error fetching OAuth2 configuration: {e!s}") from e

    def _update_oauth_settings(
        self, enterprise_url: str, oauth_config: dict[str, Any]
    ) -> None:
        try:
            config_mapping = {
                "enterprise_base_url": enterprise_url,
                "oauth2_provider": oauth_config["provider"],
                "oauth2_audience": oauth_config["audience"],
                "oauth2_client_id": oauth_config["device_authorization_client_id"],
                "oauth2_domain": oauth_config["domain"],
                "oauth2_extra": oauth_config["extra"],
            }

            console.print("🔄 Updating local OAuth2 configuration...")

            for key, value in config_mapping.items():
                self.settings_command.set(key, value)
                console.print(f"  ✓ Set {key}: {value}", style="dim")

        except Exception as e:
            raise ValueError(f"Failed to update OAuth2 settings: {e!s}") from e

    def _validate_oauth_config(self, oauth_config: dict[str, Any]) -> None:
        required_fields = [
            "audience",
            "domain",
            "device_authorization_client_id",
            "provider",
            "extra",
        ]

        missing_basic_fields = [
            field for field in required_fields if field not in oauth_config
        ]
        missing_provider_specific_fields = [
            field
            for field in self._get_provider_specific_fields(oauth_config["provider"])
            if field not in oauth_config.get("extra", {})
        ]

        if missing_basic_fields:
            raise ValueError(
                f"Missing required fields in OAuth2 configuration: [{', '.join(missing_basic_fields)}]"
            )

        if missing_provider_specific_fields:
            raise ValueError(
                f"Missing authentication provider required fields in OAuth2 configuration: [{', '.join(missing_provider_specific_fields)}] (Configured provider: '{oauth_config['provider']}')"
            )

    def _get_provider_specific_fields(self, provider_name: str) -> list[str]:
        provider = ProviderFactory.from_settings(
            Oauth2Settings(provider=provider_name, client_id="dummy", domain="dummy")
        )

        return provider.get_required_fields()

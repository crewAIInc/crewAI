import requests
from typing import Dict, Any
from rich.console import Console
from requests.exceptions import RequestException, JSONDecodeError

from crewai.cli.command import BaseCommand
from crewai.cli.settings.main import SettingsCommand
from crewai.cli.version import get_crewai_version

console = Console()


class EnterpriseConfigureCommand(BaseCommand):
    def __init__(self):
        super().__init__()
        self.settings_command = SettingsCommand()

    def configure(self, enterprise_url: str) -> None:
        try:
            enterprise_url = enterprise_url.rstrip('/')

            oauth_config = self._fetch_oauth_config(enterprise_url)

            self._update_oauth_settings(enterprise_url, oauth_config)

            console.print(
                f"✅ Successfully configured CrewAI Enterprise with OAuth2 settings from {enterprise_url}",
                style="bold green"
            )

        except Exception as e:
            console.print(f"❌ Failed to configure Enterprise settings: {str(e)}", style="bold red")
            raise SystemExit(1)

    def _fetch_oauth_config(self, enterprise_url: str) -> Dict[str, Any]:
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
            except JSONDecodeError:
                raise ValueError(f"Invalid JSON response from {oauth_endpoint}")

            required_fields = ['audience', 'domain', 'device_authorization_client_id', 'provider']
            missing_fields = [field for field in required_fields if field not in oauth_config]

            if missing_fields:
                raise ValueError(f"Missing required fields in OAuth2 configuration: {', '.join(missing_fields)}")

            console.print("✅ Successfully retrieved OAuth2 configuration", style="green")
            return oauth_config

        except RequestException as e:
            raise ValueError(f"Failed to connect to enterprise URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error fetching OAuth2 configuration: {str(e)}")

    def _update_oauth_settings(self, enterprise_url: str, oauth_config: Dict[str, Any]) -> None:
        try:
            config_mapping = {
                'enterprise_base_url': enterprise_url,
                'oauth2_provider': oauth_config['provider'],
                'oauth2_audience': oauth_config['audience'],
                'oauth2_client_id': oauth_config['device_authorization_client_id'],
                'oauth2_domain': oauth_config['domain']
            }

            console.print("🔄 Updating local OAuth2 configuration...")

            for key, value in config_mapping.items():
                self.settings_command.set(key, value)
                console.print(f"  ✓ Set {key}: {value}", style="dim")

        except Exception as e:
            raise ValueError(f"Failed to update OAuth2 settings: {str(e)}")

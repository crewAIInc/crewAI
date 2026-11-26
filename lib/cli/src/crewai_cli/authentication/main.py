"""CLI-side authentication wiring.

Re-exports the OAuth2 primitives from ``crewai_core.auth`` and overrides the
``_post_login`` hook to also log into the tool repository.
"""

from __future__ import annotations

from crewai_core.auth.oauth2 import (
    AuthenticationCommand as _BaseAuthenticationCommand,
    Oauth2Settings as Oauth2Settings,
    ProviderFactory as ProviderFactory,
    console,
)
from crewai_core.constants import DEFAULT_CREWAI_ENTERPRISE_URL
from crewai_core.settings import Settings

from crewai_cli.authentication.token import get_auth_token
from crewai_cli.plus_api import PlusAPI
from crewai_cli.tui_picker import pick


__all__ = ["AuthenticationCommand", "Oauth2Settings", "ProviderFactory"]


class AuthenticationCommand(_BaseAuthenticationCommand):
    """CLI-side login that also signs the user into the tool repository."""

    def _post_login(self) -> None:
        self._select_organization()
        self._login_to_tool_repository()

    def _select_organization(self) -> None:
        """Let the user choose the organization for this CLI login."""
        try:
            settings = Settings()
            response = PlusAPI(
                api_key=get_auth_token(),
                base_url=settings.enterprise_base_url or DEFAULT_CREWAI_ENTERPRISE_URL,
            ).get_organizations()
            response.raise_for_status()
            organizations = response.json()
        except Exception as error:
            console.print(f"Could not retrieve organizations: {error}", style="yellow")
            return

        if not organizations:
            console.print("You don't belong to any organizations yet.", style="yellow")
            return

        organization_names = [
            organization["name"] or "Unnamed organization"
            for organization in organizations
        ]
        name_width = max(len(name) for name in organization_names)
        organization_id = pick(
            "Select an organization (Name / ID):",
            [
                (organization["uuid"], organization["name"])
                for organization in organizations
            ],
            labels=[
                f"{name:<{name_width}}  {organization['uuid']}"
                for name, organization in zip(
                    organization_names, organizations, strict=True
                )
            ],
        )
        if organization_id is None:
            return

        organization = next(
            organization
            for organization in organizations
            if organization["uuid"] == organization_id
        )
        settings = Settings()
        settings.org_name = organization["name"]
        settings.org_uuid = organization["uuid"]
        settings.dump()

    def _login_to_tool_repository(self) -> None:
        from crewai_cli.tools.main import ToolCommand

        try:
            console.print(
                "Now logging you in to the Tool Repository... ",
                style="bold blue",
                end="",
            )

            settings = Settings()
            ToolCommand(
                base_url=settings.enterprise_base_url or DEFAULT_CREWAI_ENTERPRISE_URL
            ).login()

            console.print(
                "Success!\n",
                style="bold green",
            )

            settings = Settings()

            console.print(
                f"You are now authenticated to the tool repository for organization [bold cyan]'{settings.org_name if settings.org_name else settings.org_uuid}'[/bold cyan]",
                style="green",
            )
        except (Exception, SystemExit):
            console.print(
                "\n[bold yellow]Warning:[/bold yellow] Authentication with the Tool Repository failed.",
                style="yellow",
            )
            console.print(
                "Other features will work normally, but you may experience limitations "
                "with downloading and publishing tools."
                "\nRun [bold]crewai login[/bold] to try logging in again.\n",
                style="yellow",
            )

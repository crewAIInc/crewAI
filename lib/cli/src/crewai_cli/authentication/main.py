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
from crewai_core.settings import Settings


__all__ = ["AuthenticationCommand", "Oauth2Settings", "ProviderFactory"]


class AuthenticationCommand(_BaseAuthenticationCommand):
    """CLI-side login that also signs the user into the tool repository."""

    def _post_login(self) -> None:
        self._login_to_tool_repository()

    def _login_to_tool_repository(self) -> None:
        from crewai_cli.tools.main import ToolCommand

        try:
            console.print(
                "Now logging you in to the Tool Repository... ",
                style="bold blue",
                end="",
            )

            ToolCommand().login()

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

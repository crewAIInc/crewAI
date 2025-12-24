from requests import HTTPError
from rich.console import Console
from rich.table import Table

from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.cli.config import Settings


console = Console()


class OrganizationCommand(BaseCommand, PlusAPIMixin):
    def __init__(self):
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    def list(self):
        try:
            response = self.plus_api_client.get_organizations()
            response.raise_for_status()
            orgs = response.json()

            if not orgs:
                console.print(
                    "Je behoort nog tot geen organisaties.", style="yellow"
                )
                return

            table = Table(title="Jouw Organisaties")
            table.add_column("Naam", style="cyan")
            table.add_column("ID", style="green")
            for org in orgs:
                table.add_row(org["name"], org["uuid"])

            console.print(table)
        except HTTPError as e:
            if e.response.status_code == 401:
                console.print(
                    "Je bent niet ingelogd bij een organisatie. Gebruik 'crewai login' om in te loggen.",
                    style="bold red",
                )
                return
            console.print(
                f"Ophalen van organisatielijst mislukt: {e!s}", style="bold red"
            )
            raise SystemExit(1) from e
        except Exception as e:
            console.print(
                f"Ophalen van organisatielijst mislukt: {e!s}", style="bold red"
            )
            raise SystemExit(1) from e

    def switch(self, org_id):
        try:
            response = self.plus_api_client.get_organizations()
            response.raise_for_status()
            orgs = response.json()

            org = next((o for o in orgs if o["uuid"] == org_id), None)
            if not org:
                console.print(
                    f"Organisatie met id '{org_id}' niet gevonden.", style="bold red"
                )
                return

            settings = Settings()
            settings.org_name = org["name"]
            settings.org_uuid = org["uuid"]
            settings.dump()

            console.print(
                f"Succesvol gewisseld naar {org['name']} ({org['uuid']})",
                style="bold green",
            )
        except HTTPError as e:
            if e.response.status_code == 401:
                console.print(
                    "Je bent niet ingelogd bij een organisatie. Gebruik 'crewai login' om in te loggen.",
                    style="bold red",
                )
                return
            console.print(
                f"Ophalen van organisatielijst mislukt: {e!s}", style="bold red"
            )
            raise SystemExit(1) from e
        except Exception as e:
            console.print(f"Wisselen van organisatie mislukt: {e!s}", style="bold red")
            raise SystemExit(1) from e

    def current(self):
        settings = Settings()
        if settings.org_uuid:
            console.print(
                f"Momenteel ingelogd bij organisatie {settings.org_name} ({settings.org_uuid})",
                style="bold green",
            )
        else:
            console.print(
                "Je bent momenteel niet ingelogd bij een organisatie.", style="yellow"
            )
            console.print(
                "Gebruik 'crewai org list' om beschikbare organisaties te zien.", style="yellow"
            )
            console.print(
                "Gebruik 'crewai org switch <id>' om naar een organisatie te wisselen.",
                style="yellow",
            )

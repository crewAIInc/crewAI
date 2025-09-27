from rich.console import Console
from rich.table import Table

from requests import HTTPError
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
                console.print("You don't belong to any organizations yet.", style="yellow")
                return

            table = Table(title="Your Organizations")
            table.add_column("Name", style="cyan")
            table.add_column("ID", style="green")
            for org in orgs:
                table.add_row(org["name"], org["uuid"])

            console.print(table)
        except HTTPError as e:
            if e.response.status_code == 401:
                console.print("You are not logged in to any organization. Use 'crewai login' to login.", style="bold red")
                return
            console.print(f"Failed to retrieve organization list: {str(e)}", style="bold red")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"Failed to retrieve organization list: {str(e)}", style="bold red")
            raise SystemExit(1)

    def switch(self, org_id):
        try:
            response = self.plus_api_client.get_organizations()
            response.raise_for_status()
            orgs = response.json()

            org = next((o for o in orgs if o["uuid"] == org_id), None)
            if not org:
                console.print(f"Organization with id '{org_id}' not found.", style="bold red")
                return

            settings = Settings()
            settings.org_name = org["name"]
            settings.org_uuid = org["uuid"]
            settings.dump()

            console.print(f"Successfully switched to {org['name']} ({org['uuid']})", style="bold green")
        except HTTPError as e:
            if e.response.status_code == 401:
                console.print("You are not logged in to any organization. Use 'crewai login' to login.", style="bold red")
                return
            console.print(f"Failed to retrieve organization list: {str(e)}", style="bold red")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"Failed to switch organization: {str(e)}", style="bold red")
            raise SystemExit(1)

    def current(self):
        settings = Settings()
        if settings.org_uuid:
            console.print(f"Currently logged in to organization {settings.org_name} ({settings.org_uuid})", style="bold green")
        else:
            console.print("You're not currently logged in to any organization.", style="yellow")
            console.print("Use 'crewai org list' to see available organizations.", style="yellow")
            console.print("Use 'crewai org switch <id>' to switch to an organization.", style="yellow")

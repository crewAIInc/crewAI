import json
import subprocess
from typing import Any

from rich.console import Console
from rich.table import Table

from crewai.cli.command import BaseCommand, PlusAPIMixin


console = Console()


class TriggersCommand(BaseCommand, PlusAPIMixin):
    """
    A class to handle trigger-related operations for CrewAI projects.
    """

    def __init__(self):
        BaseCommand.__init__(self)
        PlusAPIMixin.__init__(self, telemetry=self._telemetry)

    def list_triggers(self) -> None:
        """List all available triggers from integrations."""
        try:
            console.print("[bold blue]Fetching available triggers...[/bold blue]")
            response = self.plus_api_client.get_triggers()
            self._validate_response(response)

            triggers_data = response.json()
            self._display_triggers(triggers_data)

        except Exception as e:
            console.print(f"[bold red]Error fetching triggers: {e}[/bold red]")
            raise SystemExit(1) from e

    def execute_with_trigger(self, trigger_path: str) -> None:
        """Execute crew with trigger payload."""
        try:
            # Parse app_slug/trigger_slug
            if "/" not in trigger_path:
                console.print(
                    "[bold red]Error: Trigger must be in format 'app_slug/trigger_slug'[/bold red]"
                )
                raise SystemExit(1)

            app_slug, trigger_slug = trigger_path.split("/", 1)

            console.print(
                f"[bold blue]Fetching trigger payload for {app_slug}/{trigger_slug}...[/bold blue]"
            )
            response = self.plus_api_client.get_trigger_payload(app_slug, trigger_slug)

            if response.status_code == 404:
                error_data = response.json()
                console.print(
                    f"[bold red]Error: {error_data.get('error', 'Trigger not found')}[/bold red]"
                )
                raise SystemExit(1)

            self._validate_response(response)

            trigger_data = response.json()
            self._display_trigger_info(trigger_data)

            # Run crew with trigger payload
            self._run_crew_with_payload(trigger_data.get("sample_payload", {}))

        except Exception as e:
            console.print(
                f"[bold red]Error executing crew with trigger: {e}[/bold red]"
            )
            raise SystemExit(1) from e

    def _display_triggers(self, triggers_data: dict[str, Any]) -> None:
        """Display triggers in a formatted table."""
        apps = triggers_data.get("apps", [])

        if not apps:
            console.print("[yellow]No triggers found.[/yellow]")
            return

        for app in apps:
            app_name = app.get("name", "Unknown App")
            app_slug = app.get("slug", "unknown")
            is_connected = app.get("is_connected", False)
            connection_status = (
                "[green]✓ Connected[/green]"
                if is_connected
                else "[red]✗ Not Connected[/red]"
            )

            console.print(
                f"\n[bold cyan]{app_name}[/bold cyan] ({app_slug}) - {connection_status}"
            )
            console.print(
                f"[dim]{app.get('description', 'No description available')}[/dim]"
            )

            triggers = app.get("triggers", [])
            if triggers:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Trigger", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Description", style="dim")

                for trigger in triggers:
                    trigger_path = f"{app_slug}/{trigger.get('slug', 'unknown')}"
                    table.add_row(
                        trigger_path,
                        trigger.get("name", "Unknown"),
                        trigger.get("description", "No description"),
                    )

                console.print(table)
            else:
                console.print("[dim]  No triggers available[/dim]")

    def _display_trigger_info(self, trigger_data: dict[str, Any]) -> None:
        """Display trigger information before execution."""
        sample_payload = trigger_data.get("sample_payload", {})
        if sample_payload:
            console.print("\n[bold yellow]Sample Payload:[/bold yellow]")
            console.print(json.dumps(sample_payload, indent=2))

    def _run_crew_with_payload(self, payload: dict[str, Any]) -> None:
        """Run the crew with the trigger payload using the run_with_trigger method."""
        try:
            subprocess.run(  # noqa: S603
                ["uv", "run", "run_with_trigger", json.dumps(payload)],  # noqa: S607
                capture_output=False,
                text=True,
                check=True,
            )

        except Exception as e:
            raise SystemExit(1) from e

import sys
import os
import subprocess
from typing import Dict, Any

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.text import Text

from crewai.cli.command import BaseCommand, PlusAPIMixin
from crewai.telemetry.telemetry import Telemetry

console = Console()


class TriggerCommand(BaseCommand, PlusAPIMixin):
    """Command handler for trigger-related operations."""

    def __init__(self):
        """Initialize the trigger command with telemetry and API client."""
        self._telemetry = Telemetry()
        super().__init__()
        PlusAPIMixin.__init__(self, self._telemetry)

    def list_triggers(self) -> None:
        """List all triggers grouped by provider name."""
        try:
            console.print("Fetching triggers from CrewAI API...", style="blue")

            # Fetch triggers from API
            response = self.plus_api_client.list_triggers()
            self._validate_response(response)

            triggers_data = response.json()

            if not triggers_data:
                console.print(
                    "No triggers found for the current user.", style="yellow"
                )
                return

            # Display triggers grouped by provider
            self._display_triggers(triggers_data)

        except requests.exceptions.ConnectionError:
            console.print(
                "Failed to connect to CrewAI API. Please check your internet connection.",
                style="bold red"
            )
            raise SystemExit(1)
        except requests.exceptions.Timeout:
            console.print(
                "Request to CrewAI API timed out. Please try again later.",
                style="bold red"
            )
            raise SystemExit(1)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                console.print(
                    "Authentication failed. Please run 'crewai login' to authenticate.",
                    style="bold red"
                )
            elif e.response.status_code == 403:
                console.print(
                    "Access denied. You may not have permission to access triggers.",
                    style="bold red"
                )
            else:
                console.print(f"HTTP error occurred: {e}", style="bold red")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"Unexpected error listing triggers: {e}", style="bold red")
            console.print("Please check your configuration and try again.", style="yellow")
            raise SystemExit(1)

    def run_trigger(self, trigger_identification: str) -> None:
        """Run a crew with the specified trigger payload."""
        try:
            # Validate trigger identification format
            if not trigger_identification or "/" not in trigger_identification:
                console.print(
                    "Invalid trigger identification format. Expected format: 'app/trigger_name'",
                    style="bold red"
                )
                console.print(
                    "Use 'crewai trigger list' to see available triggers.", style="yellow"
                )
                raise SystemExit(1)

            # Get sample payload for the trigger
            console.print(f"Getting sample payload for trigger: {trigger_identification}", style="blue")
            response = self.plus_api_client.get_trigger_sample_payload(trigger_identification)
            self._validate_response(response)

            trigger_payload = response.json()

            if not trigger_payload:
                console.print(
                    f"No sample payload found for trigger: {trigger_identification}",
                    style="yellow"
                )
                console.print(
                    "Use 'crewai trigger list' to see available triggers.", style="yellow"
                )
                return

            console.print("Sample payload retrieved successfully", style="green")

            # Import and run the crew with the trigger payload
            self._run_crew_with_payload(trigger_payload)

        except requests.exceptions.ConnectionError:
            console.print(
                "Failed to connect to CrewAI API. Please check your internet connection.",
                style="bold red"
            )
            raise SystemExit(1)
        except requests.exceptions.Timeout:
            console.print(
                "Request to CrewAI API timed out. Please try again later.",
                style="bold red"
            )
            raise SystemExit(1)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                console.print(
                    "Authentication failed. Please run 'crewai login' to authenticate.",
                    style="bold red"
                )
            elif e.response.status_code == 404:
                console.print(
                    f"Trigger '{trigger_identification}' not found.",
                    style="bold red"
                )
                console.print(
                    "Use 'crewai trigger list' to see available triggers.", style="yellow"
                )
            elif e.response.status_code == 403:
                console.print(
                    "Access denied. You may not have permission to access this trigger.",
                    style="bold red"
                )
            else:
                console.print(f"HTTP error occurred: {e}", style="bold red")
            raise SystemExit(1)
        except FileNotFoundError as e:
            console.print(
                f"Project file not found: {e}", style="bold red"
            )
            console.print(
                "Make sure you're in a valid CrewAI project directory.", style="yellow"
            )
            raise SystemExit(1)
        except subprocess.CalledProcessError as e:
            console.print(f"Error running crew: {e}", style="bold red")
            if e.output:
                console.print(f"Output: {e.output}", style="red")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"Unexpected error running trigger: {e}", style="bold red")
            console.print("Please check your configuration and try again.", style="yellow")
            raise SystemExit(1)

    def _display_triggers(self, triggers_data: Dict[str, Any]) -> None:
        """Display triggers in a formatted table grouped by provider."""
        table = Table(title="Available Triggers")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Trigger ID", style="magenta")
        table.add_column("Description", style="green")

        # Group triggers by provider
        for provider_name, triggers in triggers_data.items():
            if isinstance(triggers, dict):
                # Add provider header
                first_trigger = True

                for trigger_id, trigger_info in triggers.items():
                    description = trigger_info.get("description", "No description available")

                    # Display provider name only for the first trigger of each provider
                    provider_display = provider_name if first_trigger else ""
                    first_trigger = False

                    table.add_row(
                        provider_display,
                        trigger_id,
                        description
                    )

                # Add separator between providers (except for the last one)
                if provider_name != list(triggers_data.keys())[-1]:
                    table.add_row("", "", "")

        console.print(table)
        console.print("\nTo run a trigger, use: [bold green]crewai trigger <trigger_id>[/bold green]")

    def _run_crew_with_payload(self, trigger_payload: Dict[str, Any]) -> None:
        """Run the crew with the trigger payload."""
        script_path = None
        try:
            from crewai.cli.utils import read_toml

            # Validate project structure
            if not os.path.exists("pyproject.toml"):
                raise FileNotFoundError("pyproject.toml not found. Make sure you're in a CrewAI project directory.")

            if not os.path.exists("src"):
                raise FileNotFoundError("src directory not found. Make sure you're in a CrewAI project directory.")

            if not os.path.exists("src/main.py"):
                raise FileNotFoundError("src/main.py not found. Make sure you have a valid CrewAI project.")

            # Read project configuration
            pyproject_data = read_toml()
            is_flow = pyproject_data.get("tool", {}).get("crewai", {}).get("type") == "flow"

            console.print(f"Project type detected: {'Flow' if is_flow else 'Crew'}")
            console.print("Preparing execution environment...")

            # Create a temporary script to run the crew with trigger payload
            script_content = self._generate_crew_script(trigger_payload, is_flow)

            # Write script to temporary file
            script_path = "temp_trigger_run.py"
            with open(script_path, "w") as f:
                f.write(script_content)

            console.print(f"Running {'flow' if is_flow else 'crew'} with trigger payload...", style="blue")

            # Execute the script
            command = ["uv", "run", "python", script_path]
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            # Display success message
            console.print("✓ Execution completed successfully!", style="bold green")
            if result.stdout:
                console.print("Output:", style="blue")
                console.print(result.stdout)

        except FileNotFoundError as e:
            raise  # Re-raise to be caught by the outer try-catch
        except subprocess.CalledProcessError as e:
            error_msg = f"Crew execution failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStandard output: {e.stdout}"
            raise subprocess.CalledProcessError(e.returncode, e.cmd, error_msg)
        except Exception as e:
            raise Exception(f"Failed to execute crew: {str(e)}")
        finally:
            # Clean up temporary script
            if script_path and os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except OSError:
                    console.print(f"Warning: Could not remove temporary file {script_path}", style="yellow")

    def _generate_crew_script(self, trigger_payload: Dict[str, Any], is_flow: bool) -> str:
        """Generate a Python script to run the crew with trigger payload."""
        if is_flow:
            return f"""
import sys
sys.path.append('src')

from main import *

def main():
    try:
        # Initialize and run the flow with trigger payload
        flow = main()

        # Add trigger payload to inputs
        inputs = {{"crewai_trigger_payload": {trigger_payload}}}

        result = flow.kickoff(inputs=inputs)
        print("Flow execution completed successfully")
        print(f"Result: {{result}}")

    except Exception as e:
        print(f"Error running flow: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        else:
            return f"""
import sys
sys.path.append('src')

def main():
    try:
        # Import the crew
        from main import main as crew_main

        # Get the crew instance
        crew = crew_main()

        # Add trigger payload to inputs
        inputs = {{"crewai_trigger_payload": {trigger_payload}}}

        result = crew.kickoff(inputs=inputs)
        print("Crew execution completed successfully")
        print(f"Result: {{result}}")

    except Exception as e:
        print(f"Error running crew: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

from datetime import datetime
import os
from typing import Any

from rich.console import Console
from rich.table import Table

from crewai.cli.command import BaseCommand
from crewai.cli.config import HIDDEN_SETTINGS_KEYS, READONLY_SETTINGS_KEYS, Settings
from crewai.events.listeners.tracing.utils import _load_user_data


console = Console()


class SettingsCommand(BaseCommand):
    """A class to handle CLI configuration commands."""

    def __init__(self, settings_kwargs: dict[str, Any] | None = None):
        super().__init__()
        settings_kwargs = settings_kwargs or {}
        self.settings = Settings(**settings_kwargs)

    def list(self) -> None:
        """List all CLI configuration parameters."""
        table = Table(title="CrewAI CLI Configuration")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Description", style="yellow")

        # Add all settings to the table
        for field_name, field_info in Settings.model_fields.items():
            if field_name in HIDDEN_SETTINGS_KEYS:
                # Do not display hidden settings
                continue

            current_value = getattr(self.settings, field_name)
            description = field_info.description or "No description available"
            display_value = (
                str(current_value) if current_value not in [None, {}] else "Not set"
            )

            table.add_row(field_name, display_value, description)

        # Add trace-related settings from user data
        user_data = _load_user_data()

        # CREWAI_TRACING_ENABLED environment variable
        env_tracing = os.getenv("CREWAI_TRACING_ENABLED", "")
        env_tracing_display = env_tracing if env_tracing else "Not set"
        table.add_row(
            "CREWAI_TRACING_ENABLED",
            env_tracing_display,
            "Environment variable to enable/disable tracing",
        )

        # Trace consent status
        trace_consent = user_data.get("trace_consent")
        if trace_consent is True:
            consent_display = "✅ Enabled"
        elif trace_consent is False:
            consent_display = "❌ Disabled"
        else:
            consent_display = "Not set"
        table.add_row(
            "trace_consent", consent_display, "Whether trace collection is enabled"
        )

        # First execution timestamp
        if user_data.get("first_execution_at"):
            timestamp = datetime.fromtimestamp(user_data["first_execution_at"])
            first_exec_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            first_exec_display = "Not set"
        table.add_row(
            "first_execution_at",
            first_exec_display,
            "Timestamp of first crew/flow execution",
        )

        console.print(table)

    def set(self, key: str, value: str) -> None:
        """Set a CLI configuration parameter."""

        readonly_settings = READONLY_SETTINGS_KEYS + HIDDEN_SETTINGS_KEYS

        if not hasattr(self.settings, key) or key in readonly_settings:
            console.print(
                f"Error: Unknown or readonly configuration key '{key}'",
                style="bold red",
            )
            console.print("Available keys:", style="yellow")
            for field_name in Settings.model_fields.keys():
                if field_name not in readonly_settings:
                    console.print(f"  - {field_name}", style="yellow")
            raise SystemExit(1)

        setattr(self.settings, key, value)
        self.settings.dump()

        console.print(f"Successfully set '{key}' to '{value}'", style="bold green")

    def reset_all_settings(self) -> None:
        """Reset all CLI configuration parameters to default values."""
        self.settings.reset()
        console.print(
            "Successfully reset all configuration parameters to default values. It is recommended to run [bold yellow]'crewai login'[/bold yellow] to re-authenticate.",
            style="bold green",
        )

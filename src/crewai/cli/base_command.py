"""Base command module for CLI commands."""

import typer

class BaseCommand:
    """Base command class for CLI commands."""

    def __init__(self, app: typer.Typer = None):
        """Initialize the base command.

        Args:
            app (typer.Typer, optional): Typer app instance. Defaults to None.
        """
        self.app = app or typer.Typer()

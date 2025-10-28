"""Multion tool spec."""

import os
import subprocess
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field


class MultiOnTool(BaseTool):
    """Tool to wrap MultiOn Browse Capabilities."""

    name: str = "Multion Browse Tool"
    description: str = """Multion gives the ability for LLMs to control web browsers using natural language instructions.
            If the status is 'CONTINUE', reissue the same instruction to continue execution
        """
    multion: Any | None = None
    session_id: str | None = None
    local: bool = False
    max_steps: int = 3
    package_dependencies: list[str] = Field(default_factory=lambda: ["multion"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="MULTION_API_KEY", description="API key for Multion", required=True
            ),
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from multion.client import MultiOn  # type: ignore
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'multion' package. Would you like to install it?"
            ):
                subprocess.run(["uv", "add", "multion"], check=True)  # noqa: S607
                from multion.client import MultiOn
            else:
                raise ImportError(
                    "`multion` package not found, please run `uv add multion`"
                ) from None
        self.session_id = None
        self.multion = MultiOn(api_key=api_key or os.getenv("MULTION_API_KEY"))

    def _run(
        self,
        cmd: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Run the Multion client with the given command.

        Args:
            cmd (str): The detailed and specific natural language instructrion for web browsing

            *args (Any): Additional arguments to pass to the Multion client
            **kwargs (Any): Additional keyword arguments to pass to the Multion client
        """
        if self.multion is None:
            raise ValueError("Multion client is not initialized.")

        browse = self.multion.browse(
            cmd=cmd,
            session_id=self.session_id,
            local=self.local,
            max_steps=self.max_steps,
            *args,  # noqa: B026
            **kwargs,
        )
        self.session_id = browse.session_id

        return browse.message + "\n\n STATUS: " + browse.status

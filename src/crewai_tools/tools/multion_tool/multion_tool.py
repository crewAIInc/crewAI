"""Multion tool spec."""

from typing import Any, Optional

from crewai.tools import BaseTool


class MultiOnTool(BaseTool):
    """Tool to wrap MultiOn Browse Capabilities."""

    name: str = "Multion Browse Tool"
    description: str = """Multion gives the ability for LLMs to control web browsers using natural language instructions.
            If the status is 'CONTINUE', reissue the same instruction to continue execution
        """
    multion: Optional[Any] = None
    session_id: Optional[str] = None
    local: bool = False
    max_steps: int = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        local: bool = False,
        max_steps: int = 3,
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
                import subprocess

                subprocess.run(["uv", "add", "multion"], check=True)
                from multion.client import MultiOn
            else:
                raise ImportError(
                    "`multion` package not found, please run `uv add multion`"
                )
        self.session_id = None
        self.local = local
        self.multion = MultiOn(api_key=api_key)
        self.max_steps = max_steps

    def _run(
        self,
        cmd: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Run the Multion client with the given command.

        Args:
            cmd (str): The detailed and specific natural language instructrion for web browsing

            *args (Any): Additional arguments to pass to the Multion client
            **kwargs (Any): Additional keyword arguments to pass to the Multion client
        """

        browse = self.multion.browse(
            cmd=cmd,
            session_id=self.session_id,
            local=self.local,
            max_steps=self.max_steps,
            *args,
            **kwargs,
        )
        self.session_id = browse.session_id

        return browse.message + "\n\n STATUS: " + browse.status

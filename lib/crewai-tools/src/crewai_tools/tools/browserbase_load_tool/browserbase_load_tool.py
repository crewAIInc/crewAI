import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class BrowserbaseLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class BrowserbaseLoadTool(BaseTool):
    name: str = "Browserbase web load tool"
    description: str = "Load webpages url in a headless browser using Browserbase and return the contents"
    args_schema: type[BaseModel] = BrowserbaseLoadToolSchema
    api_key: str | None = os.getenv("BROWSERBASE_API_KEY")
    project_id: str | None = os.getenv("BROWSERBASE_PROJECT_ID")
    text_content: bool | None = False
    session_id: str | None = None
    proxy: bool | None = None
    browserbase: Any | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["browserbase"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BROWSERBASE_API_KEY",
                description="API key for Browserbase services",
                required=False,
            ),
            EnvVar(
                name="BROWSERBASE_PROJECT_ID",
                description="Project ID for Browserbase services",
                required=False,
            ),
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        text_content: bool | None = False,
        session_id: str | None = None,
        proxy: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not self.api_key:
            raise EnvironmentError(
                "BROWSERBASE_API_KEY environment variable is required for initialization"
            )
        try:
            from browserbase import Browserbase  # type: ignore
        except ImportError:
            import click

            if click.confirm(
                "`browserbase` package not found, would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "browserbase"], check=True)  # noqa: S607
                from browserbase import Browserbase  # type: ignore
            else:
                raise ImportError(
                    "`browserbase` package not found, please run `uv add browserbase`"
                ) from None

        self.browserbase = Browserbase(api_key=self.api_key)
        self.text_content = text_content
        self.session_id = session_id
        self.proxy = proxy

    def _run(self, url: str):
        return self.browserbase.load_url(  # type: ignore[union-attr]
            url, self.text_content, self.session_id, self.proxy
        )

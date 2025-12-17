import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


API_BASE_URL = "https://api.tzafon.ai"


class TzafonLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class TzafonLoadTool(BaseTool):
    """
    TzafonLoadTool
    A tool for loading and extracting content from webpages using Tzafon's browser.
    Requires `tzafon` and `playwright` packages to be installed.
    Get your API key from https://tzafon.ai/dashboard
    The Tzafon API key, can be set as an environment variable `TZAFON_API_KEY` or passed directly.

    Args:
        api_key (Optional[str]): The API key for accessing Tzafon services.
                                 If not provided, it will be retrieved from the TZAFON_API_KEY environment variable.
        **kwargs: Additional keyword arguments passed to the BaseTool constructor.
    """

    name: str = "Tzafon web load tool"
    description: str = "Load webpages url using Tzafon and return its contents"
    args_schema: type[BaseModel] = TzafonLoadToolSchema
    api_key: str | None = None
    tzafon: Any | None = None
    text_content: bool | None = True
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["tzafon", "playwright"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TZAFON_API_KEY",
                description="API key for Tzafon services",
                required=False,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        """
        Initialize the TzafonLoadTool.

        Args:
            api_key (Optional[str]): The API key for accessing Tzafon services.
                                     If not provided, it will be retrieved from the TZAFON_API_KEY environment variable.
            **kwargs: Additional keyword arguments passed to the BaseTool constructor.
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("TZAFON_API_KEY")

        if not self.api_key:
            raise EnvironmentError(
                "TZAFON_API_KEY environment variable is required for initialization"
            )

        try:
            from tzafon import Computer
        except ImportError:
            import click

            if click.confirm(
                "`tzafon` package not found, would you like to install it?"
            ):
                import subprocess

                subprocess.run(["pip", "install", "tzafon", "playwright"], check=True)  # noqa: S607
                from tzafon import Computer
            else:
                raise ImportError(
                    "`tzafon` package not found, please run `pip install tzafon`"
                ) from None

        self.tzafon = Computer(api_key=self.api_key)

    def _run(self, url: str) -> str:
        """
        Synchronously load a webpage and extract its content.

        Args:
            url (str): The URL of the webpage to load.

        Returns:
            str: The text content of the webpage if text_content is True, otherwise the HTML content.
        """
        from playwright.sync_api import sync_playwright

        computer = self.tzafon.create(kind="browser")  # type: ignore[union-attr]
        computer_id = computer.id

        with sync_playwright() as playwright:
            cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"
            browser = playwright.chromium.connect_over_cdp(cdp_url)

            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = browser.new_context()

            page = context.pages[0] if context.pages else context.new_page()

            page.goto(url)
            if self.text_content:
                page_text = page.inner_text("body")
                content = str(page_text)
            else:
                page_html = page.content()
                content = str(page_html)

            page.close()
            browser.close()

        computer.terminate()
        return content

    async def _arun(self, url: str) -> str:
        """
        Asynchronously load a webpage and extract its content.

        Args:
            url (str): The URL of the webpage to load.

        Returns:
            str: The text content of the webpage if text_content is True, otherwise the HTML content.
        """
        from playwright.async_api import async_playwright

        computer = self.tzafon.create(kind="browser")  # type: ignore[union-attr]
        computer_id = computer.id

        async with async_playwright() as playwright:
            cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"
            browser = await playwright.chromium.connect_over_cdp(cdp_url)

            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = await browser.new_context()

            page = context.pages[0] if context.pages else await context.new_page()

            await page.goto(url)
            if self.text_content:
                page_text = await page.inner_text("body")
                content = page_text
            else:
                page_html = await page.content()
                content = page_html

            await page.close()
            await browser.close()

        computer.terminate()
        return content

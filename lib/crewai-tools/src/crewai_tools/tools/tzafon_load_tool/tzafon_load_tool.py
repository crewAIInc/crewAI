import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

API_BASE_URL = "https://api.tzafon.ai"

class TzafonLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class TzafonLoadTool(BaseTool):
    name: str = "Tzafon web load tool"
    description: str = "Load webpages url using Tzafon and return its contents"
    args_schema: type[BaseModel] = TzafonLoadToolSchema
    api_key: str | None = None
    tzafon: Any | None = None
    text_content: bool | None = True
    package_dependencies: list[str] = Field(default_factory=lambda: ["tzafon", "playwright"])
    env_vars: list[EnvVar] = [
        EnvVar(name="TZAFON_API_KEY", description="API key for Tzafon services"),
    ]

    def __init__(self, api_key: str | None = None, **kwargs):
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

                subprocess.run(["pip", "install", "tzafon"], check=True)  
                from tzafon import Computer 
            else:
                raise ImportError(
                    "`tzafon` package not found, please run `pip install tzafon`"
                ) from None

        self.tzafon = Computer(api_key=self.api_key)
        self.computer = self.tzafon.create(kind="browser")

    def _run(self, url: str):
        computer_id = self.computer.id
       
        with sync_playwright() as playwright:
                cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"
                browser = playwright.chromium.connect_over_cdp(cdp_url)
                
                if browser.contexts: context = browser.contexts[0]
                else: context = browser.new_context()

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

        return content

    async def _arun(self, url: str):
        computer_id = self.computer.id
       
        async with async_playwright() as playwright:
            cdp_url = f"{API_BASE_URL}/computers/{computer_id}/cdp?token={self.api_key}"
            browser = await playwright.chromium.connect_over_cdp(cdp_url)
            
            if browser.contexts: context = browser.contexts[0]
            else: context = await browser.new_context()

            page = context.pages[0] if context.pages else await context.new_page()
            
            await page.goto(url)
            if self.text_content:
                page_text = await page.inner_text("body")
                content = page_text
            else:
                page_html = await page.content()
                content = page_html

            await page.close()
            browser.close()

        return content
        
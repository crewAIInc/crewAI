import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import validate_url


class AntibrowLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    selector: str | None = Field(
        default=None,
        description="Optional CSS selector to read instead of the whole page, for example 'main' or 'h1'",
    )


class AntibrowLoadTool(BaseTool):
    """AntibrowLoadTool.

    Load a page in a persistent AntiBrow browser profile and return its visible text.
    The profile keeps cookies, storage, an engine-level fingerprint and its own proxy
    between runs, so a page behind a login opens already signed in instead of starting
    over in a fresh browser every time.
    Requires the `antibrow` package.
    Get your API Key from https://antibrow.com/dashboard

    Args:
        api_key: The AntiBrow API key, can be set as an environment variable `ANTIBROW_API_KEY` or passed directly
        profile: Profile name. The same name always gets the same identity; unlimited and free locally
        proxy: Optional proxy URL for this profile (http, https or socks5, credentials in the URL)
        temporary: Discard the profile when the browser closes
        headless: Hide the browser window
        keep_open: Reuse one browser across calls, so a login survives between steps of a crew
        max_content_length: Truncate page text at this many characters
    """

    name: str = "AntiBrow web load tool"
    description: str = (
        "Load a web page in a persistent browser profile and return its visible text. "
        "Cookies and storage from earlier runs are kept, so pages behind a login open signed in."
    )
    args_schema: type[BaseModel] = AntibrowLoadToolSchema
    api_key: str | None = None
    profile: str = "agent"
    proxy: str | None = None
    temporary: bool = False
    headless: bool = False
    keep_open: bool = True
    max_content_length: int | None = 100000
    browser: Any | None = None
    page: Any | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["antibrow"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ANTIBROW_API_KEY",
                description="API key for the AntiBrow browser",
                required=False,
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTIBROW_API_KEY")

        try:
            from antibrow import launch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "`antibrow` package not found, please run `pip install antibrow`"
            ) from e

    def _launch(self) -> Any:
        """Launch the profile, or hand back the page already open on it."""
        if self.page is not None:
            return self.page
        from antibrow import launch

        browser = launch(
            self.profile,
            api_key=self.api_key,
            proxy=self.proxy,
            temporary=self.temporary,
            headless=self.headless,
            focus_window=False,
        )
        page = browser.new_page()
        self.browser, self.page = browser, page
        return page

    def _truncate(self, text: str) -> str:
        if self.max_content_length is None or len(text) <= self.max_content_length:
            return text
        return text[: self.max_content_length] + "\n\n[content truncated]"

    def _run(self, url: str, selector: str | None = None) -> str:
        url = validate_url(url)
        page = self._launch()
        try:
            page.goto(url, wait_until="load")
            text = page.locator(selector or "body").first.inner_text()
            return self._truncate(text)
        finally:
            if not self.keep_open:
                self.close()

    def close(self) -> None:
        """Close the browser. An abandoned one is a whole browser still running."""
        browser, self.browser, self.page = self.browser, None, None
        if browser is not None:
            browser.close()

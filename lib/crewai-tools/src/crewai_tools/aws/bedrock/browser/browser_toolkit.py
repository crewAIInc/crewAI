"""Toolkit for navigating web with AWS browser."""

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crewai_tools.aws.bedrock.browser.browser_session_manager import (
    BrowserSessionManager,
)
from crewai_tools.aws.bedrock.browser.utils import aget_current_page, get_current_page


logger = logging.getLogger(__name__)


# Input schemas
class NavigateToolInput(BaseModel):
    """Input for NavigateTool."""

    url: str = Field(description="URL to navigate to")
    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class ClickToolInput(BaseModel):
    """Input for ClickTool."""

    selector: str = Field(description="CSS selector for the element to click on")
    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""

    selector: str = Field(description="CSS selector for elements to get")
    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class ExtractTextToolInput(BaseModel):
    """Input for ExtractTextTool."""

    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class ExtractHyperlinksToolInput(BaseModel):
    """Input for ExtractHyperlinksTool."""

    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class NavigateBackToolInput(BaseModel):
    """Input for NavigateBackTool."""

    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


class CurrentWebPageToolInput(BaseModel):
    """Input for CurrentWebPageTool."""

    thread_id: str = Field(
        default="default", description="Thread ID for the browser session"
    )


# Base tool class
class BrowserBaseTool(BaseTool):
    """Base class for browser tools."""

    def __init__(self, session_manager: BrowserSessionManager):  # type: ignore[call-arg]
        """Initialize with a session manager."""
        super().__init__()  # type: ignore[call-arg]
        self._session_manager = session_manager

        if self._is_in_asyncio_loop() and hasattr(self, "_arun"):
            self._original_run = self._run

            # Override _run to use _arun when in an asyncio loop
            def patched_run(*args, **kwargs):
                try:
                    import nest_asyncio  # type: ignore[import-untyped]

                    loop = asyncio.get_event_loop()
                    nest_asyncio.apply(loop)
                    return asyncio.get_event_loop().run_until_complete(
                        self._arun(*args, **kwargs)
                    )
                except Exception as e:
                    return f"Error in patched _run: {e!s}"

            self._run = patched_run  # type: ignore[method-assign]

    async def get_async_page(self, thread_id: str) -> Any:
        """Get or create a page for the specified thread."""
        browser = await self._session_manager.get_async_browser(thread_id)
        return await aget_current_page(browser)

    def get_sync_page(self, thread_id: str) -> Any:
        """Get or create a page for the specified thread."""
        browser = self._session_manager.get_sync_browser(thread_id)
        return get_current_page(browser)

    def _is_in_asyncio_loop(self) -> bool:
        """Check if we're currently in an asyncio event loop."""
        try:
            loop = asyncio.get_event_loop()
            return loop.is_running()
        except RuntimeError:
            return False


# Tool classes
class NavigateTool(BrowserBaseTool):
    """Tool for navigating a browser to a URL."""

    name: str = "navigate_browser"
    description: str = "Navigate a browser to the specified URL"
    args_schema: type[BaseModel] = NavigateToolInput

    def _run(self, url: str, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Get page for this thread
            page = self.get_sync_page(thread_id)

            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError("URL scheme must be 'http' or 'https'")

            # Navigate to URL
            response = page.goto(url)
            status = response.status if response else "unknown"
            return f"Navigating to {url} returned status code {status}"
        except Exception as e:
            return f"Error navigating to {url}: {e!s}"

    async def _arun(self, url: str, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Get page for this thread
            page = await self.get_async_page(thread_id)

            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError("URL scheme must be 'http' or 'https'")

            # Navigate to URL
            response = await page.goto(url)
            status = response.status if response else "unknown"
            return f"Navigating to {url} returned status code {status}"
        except Exception as e:
            return f"Error navigating to {url}: {e!s}"


class ClickTool(BrowserBaseTool):
    """Tool for clicking on an element with the given CSS selector."""

    name: str = "click_element"
    description: str = "Click on an element with the given CSS selector"
    args_schema: type[BaseModel] = ClickToolInput

    visible_only: bool = True
    """Whether to consider only visible elements."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when clicking on elements."""
    playwright_timeout: float = 1_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _run(self, selector: str, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Get the current page
            page = self.get_sync_page(thread_id)

            # Click on the element
            selector_effective = self._selector_effective(selector=selector)
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

            try:
                page.click(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
            except PlaywrightTimeoutError:
                return f"Unable to click on element '{selector}'"
            except Exception as click_error:
                return f"Unable to click on element '{selector}': {click_error!s}"

            return f"Clicked element '{selector}'"
        except Exception as e:
            return f"Error clicking on element: {e!s}"

    async def _arun(self, selector: str, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Get the current page
            page = await self.get_async_page(thread_id)

            # Click on the element
            selector_effective = self._selector_effective(selector=selector)
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError

            try:
                await page.click(
                    selector_effective,
                    strict=self.playwright_strict,
                    timeout=self.playwright_timeout,
                )
            except PlaywrightTimeoutError:
                return f"Unable to click on element '{selector}'"
            except Exception as click_error:
                return f"Unable to click on element '{selector}': {click_error!s}"

            return f"Clicked element '{selector}'"
        except Exception as e:
            return f"Error clicking on element: {e!s}"


class NavigateBackTool(BrowserBaseTool):
    """Tool for navigating back in browser history."""

    name: str = "navigate_back"
    description: str = "Navigate back to the previous page"
    args_schema: type[BaseModel] = NavigateBackToolInput

    def _run(self, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Get the current page
            page = self.get_sync_page(thread_id)

            # Navigate back
            try:
                page.go_back()
                return "Navigated back to the previous page"
            except Exception as nav_error:
                return f"Unable to navigate back: {nav_error!s}"
        except Exception as e:
            return f"Error navigating back: {e!s}"

    async def _arun(self, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Get the current page
            page = await self.get_async_page(thread_id)

            # Navigate back
            try:
                await page.go_back()
                return "Navigated back to the previous page"
            except Exception as nav_error:
                return f"Unable to navigate back: {nav_error!s}"
        except Exception as e:
            return f"Error navigating back: {e!s}"


class ExtractTextTool(BrowserBaseTool):
    """Tool for extracting text from a webpage."""

    name: str = "extract_text"
    description: str = "Extract all the text on the current webpage"
    args_schema: type[BaseModel] = ExtractTextToolInput

    def _run(self, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return (
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Extract text
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\n").strip()
        except Exception as e:
            return f"Error extracting text: {e!s}"

    async def _arun(self, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return (
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Extract text
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\n").strip()
        except Exception as e:
            return f"Error extracting text: {e!s}"


class ExtractHyperlinksTool(BrowserBaseTool):
    """Tool for extracting hyperlinks from a webpage."""

    name: str = "extract_hyperlinks"
    description: str = "Extract all hyperlinks on the current webpage"
    args_schema: type[BaseModel] = ExtractHyperlinksToolInput

    def _run(self, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return (
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get the current page
            page = self.get_sync_page(thread_id)

            # Extract hyperlinks
            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            links = []
            for link in soup.find_all("a", href=True):
                text = link.get_text().strip()
                href = link["href"]
                if href.startswith(("http", "https")):  # type: ignore[union-attr]
                    links.append({"text": text, "url": href})

            if not links:
                return "No hyperlinks found on the current page."

            return json.dumps(links, indent=2)
        except Exception as e:
            return f"Error extracting hyperlinks: {e!s}"

    async def _arun(self, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return (
                    "The 'beautifulsoup4' package is required to use this tool."
                    " Please install it with 'pip install beautifulsoup4'."
                )

            # Get the current page
            page = await self.get_async_page(thread_id)

            # Extract hyperlinks
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            links = []
            for link in soup.find_all("a", href=True):
                text = link.get_text().strip()
                href = link["href"]
                if href.startswith(("http", "https")):  # type: ignore[union-attr]
                    links.append({"text": text, "url": href})

            if not links:
                return "No hyperlinks found on the current page."

            return json.dumps(links, indent=2)
        except Exception as e:
            return f"Error extracting hyperlinks: {e!s}"


class GetElementsTool(BrowserBaseTool):
    """Tool for getting elements from a webpage."""

    name: str = "get_elements"
    description: str = "Get elements from the webpage using a CSS selector"
    args_schema: type[BaseModel] = GetElementsToolInput

    def _run(self, selector: str, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Get the current page
            page = self.get_sync_page(thread_id)

            # Get elements
            elements = page.query_selector_all(selector)
            if not elements:
                return f"No elements found with selector '{selector}'"

            elements_text = []
            for i, element in enumerate(elements):
                text = element.text_content()
                elements_text.append(f"Element {i + 1}: {text.strip()}")

            return "\n".join(elements_text)
        except Exception as e:
            return f"Error getting elements: {e!s}"

    async def _arun(self, selector: str, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Get the current page
            page = await self.get_async_page(thread_id)

            # Get elements
            elements = await page.query_selector_all(selector)
            if not elements:
                return f"No elements found with selector '{selector}'"

            elements_text = []
            for i, element in enumerate(elements):
                text = await element.text_content()
                elements_text.append(f"Element {i + 1}: {text.strip()}")

            return "\n".join(elements_text)
        except Exception as e:
            return f"Error getting elements: {e!s}"


class CurrentWebPageTool(BrowserBaseTool):
    """Tool for getting information about the current webpage."""

    name: str = "current_webpage"
    description: str = "Get information about the current webpage"
    args_schema: type[BaseModel] = CurrentWebPageToolInput

    def _run(self, thread_id: str = "default", **kwargs) -> str:
        """Use the sync tool."""
        try:
            # Get the current page
            page = self.get_sync_page(thread_id)

            # Get information
            url = page.url
            title = page.title()
            return f"URL: {url}\nTitle: {title}"
        except Exception as e:
            return f"Error getting current webpage info: {e!s}"

    async def _arun(self, thread_id: str = "default", **kwargs) -> str:
        """Use the async tool."""
        try:
            # Get the current page
            page = await self.get_async_page(thread_id)

            # Get information
            url = page.url
            title = await page.title()
            return f"URL: {url}\nTitle: {title}"
        except Exception as e:
            return f"Error getting current webpage info: {e!s}"


class BrowserToolkit:
    """Toolkit for navigating web with AWS Bedrock browser.

    This toolkit provides a set of tools for working with a remote browser
    and supports multiple threads by maintaining separate browser sessions
    for each thread ID. Browsers are created lazily only when needed.

    Example:
        ```python
        from crewai import Agent, Task, Crew
        from crewai_tools.aws.bedrock.browser import create_browser_toolkit

        # Create the browser toolkit
        toolkit, browser_tools = create_browser_toolkit(region="us-west-2")

        # Create a CrewAI agent that uses the browser tools
        research_agent = Agent(
            role="Web Researcher",
            goal="Research and summarize web content",
            backstory="You're an expert at finding information online.",
            tools=browser_tools,
        )

        # Create a task for the agent
        research_task = Task(
            description="Navigate to https://example.com and extract all text content. Summarize the main points.",
            agent=research_agent,
        )

        # Create and run the crew
        crew = Crew(agents=[research_agent], tasks=[research_task])
        result = crew.kickoff()

        # Clean up browser resources when done
        import asyncio

        asyncio.run(toolkit.cleanup())
        ```
    """

    def __init__(self, region: str = "us-west-2"):
        """Initialize the toolkit.

        Args:
            region: AWS region for the browser client
        """
        self.region = region
        self.session_manager = BrowserSessionManager(region=region)
        self.tools: list[BaseTool] = []
        self._nest_current_loop()
        self._setup_tools()

    def _nest_current_loop(self):
        """Apply nest_asyncio if we're in an asyncio loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply(loop)
                except Exception as e:
                    logger.warning(f"Failed to apply nest_asyncio: {e!s}")
        except RuntimeError:
            pass

    def _setup_tools(self) -> None:
        """Initialize tools without creating any browsers."""
        self.tools = [
            NavigateTool(session_manager=self.session_manager),
            ClickTool(session_manager=self.session_manager),
            NavigateBackTool(session_manager=self.session_manager),
            ExtractTextTool(session_manager=self.session_manager),
            ExtractHyperlinksTool(session_manager=self.session_manager),
            GetElementsTool(session_manager=self.session_manager),
            CurrentWebPageTool(session_manager=self.session_manager),
        ]

    def get_tools(self) -> list[BaseTool]:
        """Get the list of browser tools.

        Returns:
            List of CrewAI tools
        """
        return self.tools

    def get_tools_by_name(self) -> dict[str, BaseTool]:
        """Get a dictionary of tools mapped by their names.

        Returns:
            Dictionary of {tool_name: tool}
        """
        return {tool.name: tool for tool in self.tools}

    async def cleanup(self) -> None:
        """Clean up all browser sessions asynchronously."""
        await self.session_manager.close_all_browsers()
        logger.info("All browser sessions cleaned up")

    def sync_cleanup(self) -> None:
        """Clean up all browser sessions from synchronous code."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.cleanup())  # noqa: RUF006
            else:
                loop.run_until_complete(self.cleanup())
        except RuntimeError:
            asyncio.run(self.cleanup())


def create_browser_toolkit(
    region: str = "us-west-2",
) -> tuple[BrowserToolkit, list[BaseTool]]:
    """Create a BrowserToolkit.

    Args:
        region: AWS region for browser client

    Returns:
        Tuple of (toolkit, tools)
    """
    toolkit = BrowserToolkit(region=region)
    tools = toolkit.get_tools()
    return toolkit, tools

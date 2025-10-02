from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from bedrock_agentcore.tools.browser_client import BrowserClient
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser

logger = logging.getLogger(__name__)


class BrowserSessionManager:
    """Manages browser sessions for different threads.

    This class maintains separate browser sessions for different threads,
    enabling concurrent usage of browsers in multi-threaded environments.
    Browsers are created lazily only when needed by tools.
    """

    def __init__(self, region: str = "us-west-2"):
        """Initialize the browser session manager.

        Args:
            region: AWS region for browser client
        """
        self.region = region
        self._async_sessions: dict[str, tuple[BrowserClient, AsyncBrowser]] = {}
        self._sync_sessions: dict[str, tuple[BrowserClient, SyncBrowser]] = {}

    async def get_async_browser(self, thread_id: str) -> AsyncBrowser:
        """Get or create an async browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            An async browser instance specific to the thread
        """
        if thread_id in self._async_sessions:
            return self._async_sessions[thread_id][1]

        return await self._create_async_browser_session(thread_id)

    def get_sync_browser(self, thread_id: str) -> SyncBrowser:
        """Get or create a sync browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            A sync browser instance specific to the thread
        """
        if thread_id in self._sync_sessions:
            return self._sync_sessions[thread_id][1]

        return self._create_sync_browser_session(thread_id)

    async def _create_async_browser_session(self, thread_id: str) -> AsyncBrowser:
        """Create a new async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The newly created async browser instance

        Raises:
            Exception: If browser session creation fails
        """
        from bedrock_agentcore.tools.browser_client import BrowserClient

        browser_client = BrowserClient(region=self.region)

        try:
            # Start browser session
            browser_client.start()

            # Get WebSocket connection info
            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to async WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.async_api import async_playwright

            # Connect to browser using Playwright
            playwright = await async_playwright().start()
            browser = await playwright.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to async browser for thread {thread_id}"
            )

            # Store session resources
            self._async_sessions[thread_id] = (browser_client, browser)

            return browser

        except Exception as e:
            logger.error(
                f"Failed to create async browser session for thread {thread_id}: {e}"
            )

            # Clean up resources if session creation fails
            if browser_client:
                try:
                    browser_client.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up browser client: {cleanup_error}")

            raise

    def _create_sync_browser_session(self, thread_id: str) -> SyncBrowser:
        """Create a new sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The newly created sync browser instance

        Raises:
            Exception: If browser session creation fails
        """
        from bedrock_agentcore.tools.browser_client import BrowserClient

        browser_client = BrowserClient(region=self.region)

        try:
            # Start browser session
            browser_client.start()

            # Get WebSocket connection info
            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to sync WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.sync_api import sync_playwright

            # Connect to browser using Playwright
            playwright = sync_playwright().start()
            browser = playwright.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to sync browser for thread {thread_id}"
            )

            # Store session resources
            self._sync_sessions[thread_id] = (browser_client, browser)

            return browser

        except Exception as e:
            logger.error(
                f"Failed to create sync browser session for thread {thread_id}: {e}"
            )

            # Clean up resources if session creation fails
            if browser_client:
                try:
                    browser_client.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up browser client: {cleanup_error}")

            raise

    async def close_async_browser(self, thread_id: str) -> None:
        """Close the async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread
        """
        if thread_id not in self._async_sessions:
            logger.warning(f"No async browser session found for thread {thread_id}")
            return

        browser_client, browser = self._async_sessions[thread_id]

        # Close browser
        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing async browser for thread {thread_id}: {e}"
                )

        # Stop browser client
        if browser_client:
            try:
                browser_client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        # Remove session from dictionary
        del self._async_sessions[thread_id]
        logger.info(f"Async browser session cleaned up for thread {thread_id}")

    def close_sync_browser(self, thread_id: str) -> None:
        """Close the sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread
        """
        if thread_id not in self._sync_sessions:
            logger.warning(f"No sync browser session found for thread {thread_id}")
            return

        browser_client, browser = self._sync_sessions[thread_id]

        # Close browser
        if browser:
            try:
                browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing sync browser for thread {thread_id}: {e}"
                )

        # Stop browser client
        if browser_client:
            try:
                browser_client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        # Remove session from dictionary
        del self._sync_sessions[thread_id]
        logger.info(f"Sync browser session cleaned up for thread {thread_id}")

    async def close_all_browsers(self) -> None:
        """Close all browser sessions."""
        # Close all async browsers
        async_thread_ids = list(self._async_sessions.keys())
        for thread_id in async_thread_ids:
            await self.close_async_browser(thread_id)

        # Close all sync browsers
        sync_thread_ids = list(self._sync_sessions.keys())
        for thread_id in sync_thread_ids:
            self.close_sync_browser(thread_id)

        logger.info("All browser sessions closed")

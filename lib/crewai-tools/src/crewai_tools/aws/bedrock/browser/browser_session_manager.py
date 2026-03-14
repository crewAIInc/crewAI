from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from bedrock_agentcore.tools.browser_client import BrowserClient
    from bedrock_agentcore.tools.config import (
        BrowserExtension,
        ProfileConfiguration,
        ProxyConfiguration,
        ViewportConfiguration,
    )
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Playwright as AsyncPlaywright
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Playwright as SyncPlaywright

logger = logging.getLogger(__name__)


class BrowserSessionManager:
    """Manages browser sessions for different threads.

    This class maintains separate browser sessions for different threads,
    enabling concurrent usage of browsers in multi-threaded environments.
    Browsers are created lazily only when needed by tools.

    Uses per-key events to serialize creation for the same thread_id without
    blocking unrelated callers or wasting resources on duplicate sessions.
    """

    def __init__(
        self,
        region: str = "us-west-2",
        identifier: str | None = None,
        session_timeout_seconds: int | None = None,
        viewport: ViewportConfiguration | dict[str, int] | None = None,
        proxy_configuration: ProxyConfiguration | dict | None = None,
        extensions: list[BrowserExtension | dict] | None = None,
        profile_configuration: ProfileConfiguration | dict | None = None,
    ):
        """Initialize the browser session manager.

        Args:
            region: AWS region for browser client
            identifier: Browser sandbox identifier. Defaults to the system browser.
                Use a custom browser ID from create_browser() for custom configurations.
            session_timeout_seconds: Session timeout in seconds (1-28800). Defaults to
                3600 (1 hour) if not specified.
            viewport: Viewport dimensions. Accepts a ``ViewportConfiguration`` instance
                (e.g. ``ViewportConfiguration.desktop_hd()``) or a plain dict
                (e.g. ``{"width": 1920, "height": 1080}``).
            proxy_configuration: Proxy routing configuration. Accepts a
                ``ProxyConfiguration`` instance or a plain dict matching the SDK shape.
            extensions: List of browser extensions. Each element can be a
                ``BrowserExtension`` instance or a plain dict matching the SDK shape.
            profile_configuration: Profile for persisting browser state across sessions.
                Accepts a ``ProfileConfiguration`` instance or a plain dict.
        """
        self.region = region
        self.identifier = identifier
        self.session_timeout_seconds = session_timeout_seconds
        self.viewport = viewport
        self.proxy_configuration = proxy_configuration
        self.extensions = extensions
        self.profile_configuration = profile_configuration
        self._lock = threading.Lock()
        self._async_sessions: dict[str, tuple[BrowserClient, AsyncBrowser]] = {}
        self._sync_sessions: dict[str, tuple[BrowserClient, SyncBrowser]] = {}
        self._async_playwrights: dict[str, AsyncPlaywright] = {}
        self._sync_playwrights: dict[str, SyncPlaywright] = {}
        self._creating: dict[str, threading.Event] = {}

    async def get_async_browser(self, thread_id: str) -> AsyncBrowser:
        """Get or create an async browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            An async browser instance specific to the thread
        """
        loop = asyncio.get_event_loop()
        while True:
            with self._lock:
                if thread_id in self._async_sessions:
                    return self._async_sessions[thread_id][1]
                if thread_id not in self._creating:
                    self._creating[thread_id] = threading.Event()
                    break
                event = self._creating[thread_id]
            ctx = contextvars.copy_context()
            await loop.run_in_executor(None, ctx.run, event.wait)

        try:
            browser_client, browser = await self._create_async_browser_session(
                thread_id
            )
            with self._lock:
                self._async_sessions[thread_id] = (browser_client, browser)
            return browser
        finally:
            with self._lock:
                evt = self._creating.pop(thread_id)
            evt.set()

    def get_sync_browser(self, thread_id: str) -> SyncBrowser:
        """Get or create a sync browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            A sync browser instance specific to the thread
        """
        while True:
            with self._lock:
                if thread_id in self._sync_sessions:
                    return self._sync_sessions[thread_id][1]
                if thread_id not in self._creating:
                    self._creating[thread_id] = threading.Event()
                    break
                event = self._creating[thread_id]
            event.wait()

        try:
            return self._create_sync_browser_session(thread_id)
        finally:
            with self._lock:
                evt = self._creating.pop(thread_id)
            evt.set()

    async def _create_async_browser_session(
        self, thread_id: str
    ) -> tuple[BrowserClient, AsyncBrowser]:
        """Create a new async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            Tuple of (BrowserClient, AsyncBrowser).

        Raises:
            Exception: If browser session creation fails
        """
        from bedrock_agentcore.tools.browser_client import BrowserClient

        browser_client = BrowserClient(
            region=self.region,
            integration_source="crewai",
        )

        try:
            start_kwargs: dict[str, Any] = {}
            if self.identifier is not None:
                start_kwargs["identifier"] = self.identifier
            if self.session_timeout_seconds is not None:
                start_kwargs["session_timeout_seconds"] = self.session_timeout_seconds
            if self.viewport is not None:
                start_kwargs["viewport"] = self.viewport
            if self.proxy_configuration is not None:
                start_kwargs["proxy_configuration"] = self.proxy_configuration
            if self.extensions is not None:
                start_kwargs["extensions"] = self.extensions
            if self.profile_configuration is not None:
                start_kwargs["profile_configuration"] = self.profile_configuration
            browser_client.start(**start_kwargs)

            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to async WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.async_api import async_playwright

            pw = await async_playwright().start()
            browser = await pw.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to async browser for thread {thread_id}"
            )

            with self._lock:
                self._async_playwrights[thread_id] = pw

            return browser_client, browser

        except Exception as e:
            logger.error(
                f"Failed to create async browser session for thread {thread_id}: {e}"
            )

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

        browser_client = BrowserClient(
            region=self.region,
            integration_source="crewai",
        )

        try:
            start_kwargs: dict[str, Any] = {}
            if self.identifier is not None:
                start_kwargs["identifier"] = self.identifier
            if self.session_timeout_seconds is not None:
                start_kwargs["session_timeout_seconds"] = self.session_timeout_seconds
            if self.viewport is not None:
                start_kwargs["viewport"] = self.viewport
            if self.proxy_configuration is not None:
                start_kwargs["proxy_configuration"] = self.proxy_configuration
            if self.extensions is not None:
                start_kwargs["extensions"] = self.extensions
            if self.profile_configuration is not None:
                start_kwargs["profile_configuration"] = self.profile_configuration
            browser_client.start(**start_kwargs)

            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to sync WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            browser = pw.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to sync browser for thread {thread_id}"
            )

            with self._lock:
                self._sync_playwrights[thread_id] = pw
                self._sync_sessions[thread_id] = (browser_client, browser)

            return browser

        except Exception as e:
            logger.error(
                f"Failed to create sync browser session for thread {thread_id}: {e}"
            )

            if browser_client:
                try:
                    browser_client.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up browser client: {cleanup_error}")

            raise

    def reconnect_sync_browser(self, thread_id: str) -> SyncBrowser:
        """Reconnect the Playwright browser for a sync session.

        This is needed after release_control() which invalidates the CDP
        connection while keeping the BrowserClient session alive.  Reuses the
        existing Playwright instance to avoid "Sync API inside asyncio loop"
        errors from starting a second playwright subprocess.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The reconnected sync browser instance

        Raises:
            RuntimeError: If no session exists for the thread
        """
        with self._lock:
            if thread_id not in self._sync_sessions:
                raise RuntimeError(f"No sync session for thread {thread_id}")
            browser_client, old_browser = self._sync_sessions[thread_id]
            pw = self._sync_playwrights.get(thread_id)

        # Close old Playwright connection (may already be dead)
        try:
            old_browser.close()
        except Exception:
            pass

        # Reconnect using existing browser_client (session still active)
        ws_url, headers = browser_client.generate_ws_headers()
        logger.info(
            f"Reconnecting sync browser for thread {thread_id}: {ws_url}"
        )

        if pw is None:
            from playwright.sync_api import sync_playwright

            pw = sync_playwright().start()
            with self._lock:
                self._sync_playwrights[thread_id] = pw

        browser = pw.chromium.connect_over_cdp(
            endpoint_url=ws_url, headers=headers, timeout=30000
        )
        logger.info(f"Successfully reconnected sync browser for thread {thread_id}")

        with self._lock:
            self._sync_sessions[thread_id] = (browser_client, browser)

        return browser

    async def reconnect_async_browser(self, thread_id: str) -> AsyncBrowser:
        """Reconnect the Playwright browser for an async session.

        This is needed after release_control() which invalidates the CDP
        connection while keeping the BrowserClient session alive.  Reuses the
        existing Playwright instance to avoid creating a duplicate subprocess.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The reconnected async browser instance

        Raises:
            RuntimeError: If no session exists for the thread
        """
        with self._lock:
            if thread_id not in self._async_sessions:
                raise RuntimeError(f"No async session for thread {thread_id}")
            browser_client, old_browser = self._async_sessions[thread_id]
            pw = self._async_playwrights.get(thread_id)

        try:
            await old_browser.close()
        except Exception:
            pass

        ws_url, headers = browser_client.generate_ws_headers()
        logger.info(
            f"Reconnecting async browser for thread {thread_id}: {ws_url}"
        )

        if pw is None:
            from playwright.async_api import async_playwright

            pw = await async_playwright().start()
            with self._lock:
                self._async_playwrights[thread_id] = pw

        browser = await pw.chromium.connect_over_cdp(
            endpoint_url=ws_url, headers=headers, timeout=30000
        )
        logger.info(f"Successfully reconnected async browser for thread {thread_id}")

        with self._lock:
            self._async_sessions[thread_id] = (browser_client, browser)

        return browser

    def get_browser_client(self, thread_id: str) -> BrowserClient | None:
        """Get the BrowserClient for a sync session, or None if not started."""
        with self._lock:
            if thread_id in self._sync_sessions:
                return self._sync_sessions[thread_id][0]
        return None

    async def get_async_browser_client(self, thread_id: str) -> BrowserClient | None:
        """Get the BrowserClient for an async session, or None if not started."""
        with self._lock:
            if thread_id in self._async_sessions:
                return self._async_sessions[thread_id][0]
        return None

    async def close_async_browser(self, thread_id: str) -> None:
        """Close the async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread
        """
        with self._lock:
            if thread_id not in self._async_sessions:
                logger.warning(f"No async browser session found for thread {thread_id}")
                return

            browser_client, browser = self._async_sessions.pop(thread_id)
            pw = self._async_playwrights.pop(thread_id, None)

        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing async browser for thread {thread_id}: {e}"
                )

        if pw:
            try:
                await pw.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping async playwright for thread {thread_id}: {e}"
                )

        if browser_client:
            try:
                browser_client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        logger.info(f"Async browser session cleaned up for thread {thread_id}")

    def close_sync_browser(self, thread_id: str) -> None:
        """Close the sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread
        """
        with self._lock:
            if thread_id not in self._sync_sessions:
                logger.warning(f"No sync browser session found for thread {thread_id}")
                return

            browser_client, browser = self._sync_sessions.pop(thread_id)
            pw = self._sync_playwrights.pop(thread_id, None)

        if browser:
            try:
                browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing sync browser for thread {thread_id}: {e}"
                )

        if pw:
            try:
                pw.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping sync playwright for thread {thread_id}: {e}"
                )

        if browser_client:
            try:
                browser_client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        logger.info(f"Sync browser session cleaned up for thread {thread_id}")

    async def close_all_browsers(self) -> None:
        """Close all browser sessions."""
        with self._lock:
            async_thread_ids = list(self._async_sessions.keys())
            sync_thread_ids = list(self._sync_sessions.keys())

        for thread_id in async_thread_ids:
            await self.close_async_browser(thread_id)

        for thread_id in sync_thread_ids:
            self.close_sync_browser(thread_id)

        logger.info("All browser sessions closed")

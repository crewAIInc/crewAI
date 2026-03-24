"""Unit tests for AWS Bedrock AgentCore Browser toolkit."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from crewai_tools.aws.bedrock.browser.browser_session_manager import (
    BrowserSessionManager,
)
from crewai_tools.aws.bedrock.browser.browser_toolkit import (
    BrowserToolkit,
    GenerateLiveViewUrlTool,
    TakeControlTool,
    ReleaseControlTool,
    create_browser_toolkit,
)


# --- BrowserSessionManager ---


class TestBrowserSessionManager:
    def test_init_defaults(self):
        mgr = BrowserSessionManager(region="us-east-1")
        assert mgr.region == "us-east-1"
        assert mgr.identifier is None

    def test_init_with_identifier(self):
        mgr = BrowserSessionManager(region="us-west-2", identifier="vpc-browser")
        assert mgr.identifier == "vpc-browser"

    def test_integration_source_set_sync(self):
        """Verify integration_source='crewai' is passed to BrowserClient in sync path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp.return_value = mock_browser

        fake_playwright_module = MagicMock()
        mock_sync_pw = MagicMock()
        mock_sync_pw.return_value.start.return_value = mock_playwright_ctx
        fake_playwright_module.sync_playwright = mock_sync_pw

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.sync_api": fake_playwright_module,
        }):
            mgr = BrowserSessionManager(region="us-west-2")
            mgr._create_sync_browser_session("thread-1")

        MockBC.assert_called_once_with(
            region="us-west-2",
            integration_source="crewai",
        )
        mock_bc.start.assert_called_once_with()

    def test_identifier_passed_to_start_sync(self):
        """Verify identifier is passed to start() in sync path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp.return_value = mock_browser

        fake_playwright_module = MagicMock()
        mock_sync_pw = MagicMock()
        mock_sync_pw.return_value.start.return_value = mock_playwright_ctx
        fake_playwright_module.sync_playwright = mock_sync_pw

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.sync_api": fake_playwright_module,
        }):
            mgr = BrowserSessionManager(
                region="us-west-2", identifier="my-vpc-browser"
            )
            mgr._create_sync_browser_session("thread-1")

        mock_bc.start.assert_called_once_with(identifier="my-vpc-browser")

    def test_session_config_passed_to_start_sync(self):
        """Verify all session config is passed to start() in sync path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp.return_value = mock_browser

        fake_playwright_module = MagicMock()
        mock_sync_pw = MagicMock()
        mock_sync_pw.return_value.start.return_value = mock_playwright_ctx
        fake_playwright_module.sync_playwright = mock_sync_pw

        viewport = {"width": 1920, "height": 1080}
        proxy = {"proxies": [{"externalProxy": {"server": "p.co", "port": 8080}}]}
        extensions = [{"location": {"s3": {"bucket": "b", "prefix": "e/"}}}]
        profile = {"profileIdentifier": "prof-1"}

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.sync_api": fake_playwright_module,
        }):
            mgr = BrowserSessionManager(
                region="us-west-2",
                identifier="custom-id",
                session_timeout_seconds=7200,
                viewport=viewport,
                proxy_configuration=proxy,
                extensions=extensions,
                profile_configuration=profile,
            )
            mgr._create_sync_browser_session("thread-1")

        mock_bc.start.assert_called_once_with(
            identifier="custom-id",
            session_timeout_seconds=7200,
            viewport=viewport,
            proxy_configuration=proxy,
            extensions=extensions,
            profile_configuration=profile,
        )

    def test_session_config_typed_dataclasses_passed_to_start_sync(self):
        """Verify SDK typed dataclasses are passed through to start() in sync path."""
        import sys

        from bedrock_agentcore.tools.config import (
            BrowserExtension,
            ExtensionS3Location,
            ExternalProxy,
            ProfileConfiguration,
            ProxyConfiguration,
            ViewportConfiguration,
        )

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp.return_value = mock_browser

        fake_playwright_module = MagicMock()
        mock_sync_pw = MagicMock()
        mock_sync_pw.return_value.start.return_value = mock_playwright_ctx
        fake_playwright_module.sync_playwright = mock_sync_pw

        viewport = ViewportConfiguration.desktop_hd()
        proxy = ProxyConfiguration(
            proxies=[ExternalProxy(server="proxy.co", port=8080)],
            bypass_patterns=[".internal.com"],
        )
        extensions = [
            BrowserExtension(
                s3_location=ExtensionS3Location(bucket="ext-bucket", prefix="ublock/")
            )
        ]
        profile = ProfileConfiguration(profile_identifier="my-profile")

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.sync_api": fake_playwright_module,
        }):
            mgr = BrowserSessionManager(
                region="us-west-2",
                viewport=viewport,
                proxy_configuration=proxy,
                extensions=extensions,
                profile_configuration=profile,
            )
            mgr._create_sync_browser_session("thread-1")

        mock_bc.start.assert_called_once_with(
            viewport=viewport,
            proxy_configuration=proxy,
            extensions=extensions,
            profile_configuration=profile,
        )

    def test_get_browser_client_returns_none_when_no_session(self):
        mgr = BrowserSessionManager(region="us-west-2")
        assert mgr.get_browser_client("nonexistent") is None

    def test_get_browser_client_returns_client(self):
        mgr = BrowserSessionManager(region="us-west-2")
        mock_client = MagicMock()
        mock_browser = MagicMock()
        mgr._sync_sessions["thread-1"] = (mock_client, mock_browser)

        result = mgr.get_browser_client("thread-1")
        assert result is mock_client

    @pytest.mark.asyncio
    async def test_get_async_browser_client_returns_none(self):
        mgr = BrowserSessionManager(region="us-west-2")
        result = await mgr.get_async_browser_client("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_async_browser_client_returns_client(self):
        mgr = BrowserSessionManager(region="us-west-2")
        mock_client = MagicMock()
        mock_browser = MagicMock()
        mgr._async_sessions["thread-1"] = (mock_client, mock_browser)

        result = await mgr.get_async_browser_client("thread-1")
        assert result is mock_client

    @pytest.mark.asyncio
    async def test_integration_source_set_async(self):
        """Verify integration_source='crewai' is passed to BrowserClient in async path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

        fake_async_pw_module = MagicMock()
        mock_async_pw = MagicMock()
        mock_async_pw_start = AsyncMock(return_value=mock_playwright_ctx)
        mock_async_pw.return_value.start = mock_async_pw_start
        fake_async_pw_module.async_playwright = mock_async_pw

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.async_api": fake_async_pw_module,
        }):
            mgr = BrowserSessionManager(region="us-west-2")
            await mgr._create_async_browser_session("thread-1")

        MockBC.assert_called_once_with(
            region="us-west-2",
            integration_source="crewai",
        )
        mock_bc.start.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_identifier_passed_to_start_async(self):
        """Verify identifier is passed to start() in async path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

        fake_async_pw_module = MagicMock()
        mock_async_pw = MagicMock()
        mock_async_pw_start = AsyncMock(return_value=mock_playwright_ctx)
        mock_async_pw.return_value.start = mock_async_pw_start
        fake_async_pw_module.async_playwright = mock_async_pw

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.async_api": fake_async_pw_module,
        }):
            mgr = BrowserSessionManager(
                region="us-west-2", identifier="my-vpc-browser"
            )
            await mgr._create_async_browser_session("thread-1")

        mock_bc.start.assert_called_once_with(identifier="my-vpc-browser")

    @pytest.mark.asyncio
    async def test_session_config_passed_to_start_async(self):
        """Verify all session config is passed to start() in async path."""
        import sys

        MockBC = MagicMock()
        mock_bc = MagicMock()
        mock_bc.generate_ws_headers.return_value = ("ws://fake", {})
        MockBC.return_value = mock_bc

        fake_module = MagicMock()
        fake_module.BrowserClient = MockBC

        mock_playwright_ctx = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_ctx.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

        fake_async_pw_module = MagicMock()
        mock_async_pw = MagicMock()
        mock_async_pw_start = AsyncMock(return_value=mock_playwright_ctx)
        mock_async_pw.return_value.start = mock_async_pw_start
        fake_async_pw_module.async_playwright = mock_async_pw

        viewport = {"width": 375, "height": 667}
        proxy = {"proxies": []}
        extensions = [{"location": {"s3": {"bucket": "b", "prefix": "e/"}}}]
        profile = {"profileIdentifier": "mobile-prof"}

        with patch.dict(sys.modules, {
            "bedrock_agentcore.tools.browser_client": fake_module,
            "playwright": MagicMock(),
            "playwright.async_api": fake_async_pw_module,
        }):
            mgr = BrowserSessionManager(
                region="us-west-2",
                identifier="custom-id",
                session_timeout_seconds=1800,
                viewport=viewport,
                proxy_configuration=proxy,
                extensions=extensions,
                profile_configuration=profile,
            )
            await mgr._create_async_browser_session("thread-1")

        mock_bc.start.assert_called_once_with(
            identifier="custom-id",
            session_timeout_seconds=1800,
            viewport=viewport,
            proxy_configuration=proxy,
            extensions=extensions,
            profile_configuration=profile,
        )


# --- BrowserToolkit ---


class TestBrowserToolkit:
    @patch(
        "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
    )
    def test_tool_count(self, MockMgr):
        """Verify all 10 tools are registered."""
        toolkit = BrowserToolkit(region="us-west-2")
        assert len(toolkit.tools) == 10

    @patch(
        "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
    )
    def test_identifier_passed_to_session_manager(self, MockMgr):
        """Verify identifier is forwarded to BrowserSessionManager."""
        BrowserToolkit(region="us-west-2", identifier="my-id")
        MockMgr.assert_called_once_with(
            region="us-west-2",
            identifier="my-id",
            session_timeout_seconds=None,
            viewport=None,
            proxy_configuration=None,
            extensions=None,
            profile_configuration=None,
        )

    @patch(
        "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
    )
    def test_session_config_passed_to_session_manager(self, MockMgr):
        """Verify all session config is forwarded to BrowserSessionManager."""
        viewport = {"width": 1920, "height": 1080}
        proxy = {"proxies": [{"externalProxy": {"server": "p.co", "port": 8080}}]}
        extensions = [{"location": {"s3": {"bucket": "b", "prefix": "e/"}}}]
        profile = {"profileIdentifier": "prof-1"}

        BrowserToolkit(
            region="eu-west-1",
            identifier="custom-id",
            session_timeout_seconds=7200,
            viewport=viewport,
            proxy_configuration=proxy,
            extensions=extensions,
            profile_configuration=profile,
        )
        MockMgr.assert_called_once_with(
            region="eu-west-1",
            identifier="custom-id",
            session_timeout_seconds=7200,
            viewport=viewport,
            proxy_configuration=proxy,
            extensions=extensions,
            profile_configuration=profile,
        )

    @patch(
        "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
    )
    def test_tool_names(self, MockMgr):
        """Verify expected tool names are present."""
        toolkit = BrowserToolkit(region="us-west-2")
        names = {t.name for t in toolkit.tools}
        expected = {
            "navigate_browser",
            "click_element",
            "navigate_back",
            "extract_text",
            "extract_hyperlinks",
            "get_elements",
            "current_webpage",
            "generate_live_view_url",
            "take_control",
            "release_control",
        }
        assert names == expected

    def test_create_browser_toolkit_passes_identifier(self):
        with patch(
            "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
        ):
            toolkit, tools = create_browser_toolkit(
                region="eu-west-1", identifier="vpc-id"
            )
        assert len(tools) == 10

    @patch(
        "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
    )
    def test_session_config_typed_dataclasses_passed_to_session_manager(self, MockMgr):
        """Verify SDK typed dataclasses are forwarded to BrowserSessionManager."""
        from bedrock_agentcore.tools.config import (
            BrowserExtension,
            ExtensionS3Location,
            ProfileConfiguration,
            ViewportConfiguration,
        )

        viewport = ViewportConfiguration(width=1280, height=720)
        extensions = [
            BrowserExtension(
                s3_location=ExtensionS3Location(bucket="b", prefix="e/")
            )
        ]
        profile = ProfileConfiguration(profile_identifier="prof-1")

        BrowserToolkit(
            region="us-west-2",
            viewport=viewport,
            extensions=extensions,
            profile_configuration=profile,
        )
        MockMgr.assert_called_once_with(
            region="us-west-2",
            identifier=None,
            session_timeout_seconds=None,
            viewport=viewport,
            proxy_configuration=None,
            extensions=extensions,
            profile_configuration=profile,
        )

    def test_create_browser_toolkit_passes_session_config(self):
        with patch(
            "crewai_tools.aws.bedrock.browser.browser_toolkit.BrowserSessionManager"
        ) as MockMgr:
            create_browser_toolkit(
                region="us-west-2",
                session_timeout_seconds=3600,
                viewport={"width": 800, "height": 600},
                proxy_configuration={"proxies": []},
                extensions=[],
                profile_configuration={"profileIdentifier": "p1"},
            )
        MockMgr.assert_called_once_with(
            region="us-west-2",
            identifier=None,
            session_timeout_seconds=3600,
            viewport={"width": 800, "height": 600},
            proxy_configuration={"proxies": []},
            extensions=[],
            profile_configuration={"profileIdentifier": "p1"},
        )


# --- GenerateLiveViewUrlTool ---


class TestGenerateLiveViewUrlTool:
    def test_run_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.generate_live_view_url.return_value = "https://live-view.example.com/abc"
        mock_mgr.get_browser_client.return_value = mock_client

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")

        mock_mgr.get_browser_client.assert_called_once_with("default")
        mock_client.generate_live_view_url.assert_called_once()
        assert "https://live-view.example.com/abc" in result

    def test_run_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_browser_client.return_value = None

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "No browser session found" in result

    def test_run_error(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.generate_live_view_url.side_effect = RuntimeError("auth failed")
        mock_mgr.get_browser_client.return_value = mock_client

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "Error generating live view URL" in result

    @pytest.mark.asyncio
    async def test_arun_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.generate_live_view_url.return_value = "https://live.example.com"
        mock_mgr.get_async_browser_client = AsyncMock(return_value=mock_client)

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")
        assert "https://live.example.com" in result

    @pytest.mark.asyncio
    async def test_arun_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_async_browser_client = AsyncMock(return_value=None)

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")
        assert "No browser session found" in result

    @pytest.mark.asyncio
    async def test_arun_error(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.generate_live_view_url.side_effect = RuntimeError("auth failed")
        mock_mgr.get_async_browser_client = AsyncMock(return_value=mock_client)

        tool = GenerateLiveViewUrlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")
        assert "Error generating live view URL" in result


# --- TakeControlTool ---


class TestTakeControlTool:
    def test_run_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get_browser_client.return_value = mock_client

        tool = TakeControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")

        mock_client.take_control.assert_called_once()
        assert "Manual control enabled" in result

    def test_run_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_browser_client.return_value = None

        tool = TakeControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "No browser session found" in result

    def test_run_error(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.take_control.side_effect = RuntimeError("control failed")
        mock_mgr.get_browser_client.return_value = mock_client

        tool = TakeControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "Error taking control" in result

    @pytest.mark.asyncio
    async def test_arun_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get_async_browser_client = AsyncMock(return_value=mock_client)

        tool = TakeControlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")

        mock_client.take_control.assert_called_once()
        assert "Manual control enabled" in result

    @pytest.mark.asyncio
    async def test_arun_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_async_browser_client = AsyncMock(return_value=None)

        tool = TakeControlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")
        assert "No browser session found" in result


# --- ReleaseControlTool ---


class TestReleaseControlTool:
    def test_run_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get_browser_client.return_value = mock_client

        tool = ReleaseControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")

        mock_client.release_control.assert_called_once()
        mock_mgr.reconnect_sync_browser.assert_called_once_with("default")
        assert "re-enabled" in result

    def test_run_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_browser_client.return_value = None

        tool = ReleaseControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "No browser session found" in result

    def test_run_error(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_client.release_control.side_effect = RuntimeError("release failed")
        mock_mgr.get_browser_client.return_value = mock_client

        tool = ReleaseControlTool(session_manager=mock_mgr)
        result = tool._run(thread_id="default")
        assert "Error releasing control" in result

    @pytest.mark.asyncio
    async def test_arun_success(self):
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get_async_browser_client = AsyncMock(return_value=mock_client)
        mock_mgr.reconnect_async_browser = AsyncMock()

        tool = ReleaseControlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")

        mock_client.release_control.assert_called_once()
        mock_mgr.reconnect_async_browser.assert_called_once_with("t1")
        assert "re-enabled" in result

    @pytest.mark.asyncio
    async def test_arun_no_session(self):
        mock_mgr = MagicMock()
        mock_mgr.get_async_browser_client = AsyncMock(return_value=None)

        tool = ReleaseControlTool(session_manager=mock_mgr)
        result = await tool._arun(thread_id="t1")
        assert "No browser session found" in result

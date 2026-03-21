"""Tests for MCP security integration (issue #4875).

Covers:
- MCPSecurityConfig model validation
- MCPSecurityManager key generation, passport creation, message signing,
  tool integrity verification, replay protection, and graceful degradation
- Integration with MCPClient and MCPToolResolver via security_manager
- Config models accepting optional security field
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.mcp.config import MCPServerHTTP, MCPServerSSE, MCPServerStdio
from crewai.mcp.security import MCPSecurityConfig, MCPSecurityManager


# ---------------------------------------------------------------------------
# MCPSecurityConfig tests
# ---------------------------------------------------------------------------


class TestMCPSecurityConfig:
    """Tests for the MCPSecurityConfig Pydantic model."""

    def test_default_values(self):
        config = MCPSecurityConfig()
        assert config.agent_name == "crewai-agent"
        assert config.agent_version == "1.0.0"
        assert config.capabilities == []
        assert config.private_key is None
        assert config.public_key is None
        assert config.ta_private_key is None
        assert config.ta_public_key is None
        assert config.sign_messages is True
        assert config.verify_tool_signatures is True

    def test_custom_values(self):
        config = MCPSecurityConfig(
            agent_name="researcher",
            agent_version="2.0.0",
            capabilities=["search", "summarize"],
            sign_messages=False,
            verify_tool_signatures=False,
        )
        assert config.agent_name == "researcher"
        assert config.agent_version == "2.0.0"
        assert config.capabilities == ["search", "summarize"]
        assert config.sign_messages is False
        assert config.verify_tool_signatures is False


# ---------------------------------------------------------------------------
# MCPSecurityManager tests
# ---------------------------------------------------------------------------


class TestMCPSecurityManager:
    """Tests for the MCPSecurityManager class."""

    def test_is_available(self):
        """mcp-secure is installed in the test env so this should be True."""
        assert MCPSecurityManager.is_available() is True

    def test_auto_generates_keys(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        assert manager.public_key is not None
        assert manager._private_key is not None
        assert manager.passport is not None
        assert manager.passport_id is not None

    def test_uses_provided_keys(self):
        from mcp_secure import generate_key_pair

        keys = generate_key_pair()
        config = MCPSecurityConfig(
            agent_name="test-agent",
            private_key=keys["private_key"],
            public_key=keys["public_key"],
        )
        manager = MCPSecurityManager(config)

        assert manager._private_key == keys["private_key"]
        assert manager.public_key == keys["public_key"]

    def test_passport_creation(self):
        config = MCPSecurityConfig(
            agent_name="researcher",
            agent_version="2.0.0",
            capabilities=["search"],
        )
        manager = MCPSecurityManager(config)
        passport = manager.passport

        assert passport is not None
        assert passport["agent"]["name"] == "researcher"
        assert passport["agent"]["version"] == "2.0.0"
        assert "search" in passport["agent"]["capabilities"]

    def test_passport_signed_by_ta(self):
        from mcp_secure import generate_key_pair, verify_passport_signature

        ta_keys = generate_key_pair()
        config = MCPSecurityConfig(
            agent_name="test-agent",
            ta_private_key=ta_keys["private_key"],
            ta_public_key=ta_keys["public_key"],
        )
        manager = MCPSecurityManager(config)

        assert manager.passport is not None
        assert "signature" in manager.passport
        assert verify_passport_signature(manager.passport, ta_keys["public_key"])

    # -- Message signing / verification ------------------------------------

    def test_sign_message(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        message = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {"name": "read_file", "arguments": {"path": "/tmp/test"}},
        }
        envelope = manager.sign_message(message)

        # Envelope must contain MCPS fields
        assert "mcps" in envelope
        assert "passport_id" in envelope
        assert "timestamp" in envelope
        assert "nonce" in envelope
        assert "signature" in envelope
        assert "message" in envelope

    def test_sign_message_disabled(self):
        config = MCPSecurityConfig(agent_name="test-agent", sign_messages=False)
        manager = MCPSecurityManager(config)

        message = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        result = manager.sign_message(message)

        # Should return the original message unchanged
        assert result is message

    def test_verify_own_message(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
        envelope = manager.sign_message(message)

        assert manager.verify_message(envelope, manager.public_key)

    def test_verify_tampered_message_fails(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
        envelope = manager.sign_message(message)

        # Tamper with the message inside the envelope
        envelope["message"]["method"] = "tools/evil"

        assert manager.verify_message(envelope, manager.public_key) is False

    def test_replay_protection(self):
        """Same nonce used twice should be detected as replay."""
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
        envelope = manager.sign_message(message)

        # First verification should pass
        assert manager.verify_message(envelope, manager.public_key) is True
        # Second verification with same nonce should fail (replay)
        assert manager.verify_message(envelope, manager.public_key) is False

    def test_unique_nonces(self):
        """Each signed message should have a unique nonce."""
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
        envelope1 = manager.sign_message(message)
        envelope2 = manager.sign_message(message)

        assert envelope1["nonce"] != envelope2["nonce"]

    # -- Tool integrity ----------------------------------------------------

    def test_sign_and_verify_tool(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        tool = {
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
        signature = manager.sign_tool(tool)
        assert signature is not None
        assert manager.verify_tool(tool, signature) is True

    def test_verify_tampered_tool_fails(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        tool = {
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {"type": "object"},
        }
        signature = manager.sign_tool(tool)

        # Tamper with the tool definition
        tool["description"] = "EVIL: delete all files"
        assert manager.verify_tool(tool, signature) is False

    def test_verify_tool_disabled(self):
        config = MCPSecurityConfig(
            agent_name="test-agent", verify_tool_signatures=False
        )
        manager = MCPSecurityManager(config)

        tool = {"name": "anything", "description": "whatever"}
        # Should always return True when verification is disabled
        assert manager.verify_tool(tool, "invalid-signature") is True

    # -- Passport verification ---------------------------------------------

    def test_verify_passport(self):
        from mcp_secure import generate_key_pair

        ta_keys = generate_key_pair()
        config = MCPSecurityConfig(
            agent_name="test-agent",
            ta_private_key=ta_keys["private_key"],
            ta_public_key=ta_keys["public_key"],
        )
        manager = MCPSecurityManager(config)

        assert manager.verify_passport(manager.passport, ta_keys["public_key"]) is True

    # -- Cleanup -----------------------------------------------------------

    def test_destroy(self):
        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        assert manager.passport is not None
        manager.destroy()
        assert manager.passport is None

    # -- Graceful degradation when mcp-secure not installed ----------------

    def test_graceful_degradation_sign_message(self):
        """sign_message returns the original message when mcp-secure is unavailable."""
        config = MCPSecurityConfig(agent_name="test-agent")

        with patch("crewai.mcp.security._mcp_secure_available", return_value=False):
            manager = MCPSecurityManager(config)
            message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
            result = manager.sign_message(message)
            assert result is message

    def test_graceful_degradation_verify_message(self):
        """verify_message returns True when mcp-secure is unavailable."""
        config = MCPSecurityConfig(agent_name="test-agent")

        with patch("crewai.mcp.security._mcp_secure_available", return_value=False):
            manager = MCPSecurityManager(config)
            assert manager.verify_message({}, "any-key") is True

    def test_graceful_degradation_verify_tool(self):
        """verify_tool returns True when mcp-secure is unavailable."""
        config = MCPSecurityConfig(agent_name="test-agent")

        with patch("crewai.mcp.security._mcp_secure_available", return_value=False):
            manager = MCPSecurityManager(config)
            assert manager.verify_tool({"name": "x"}, "sig") is True

    def test_graceful_degradation_sign_tool(self):
        """sign_tool returns None when mcp-secure is unavailable."""
        config = MCPSecurityConfig(agent_name="test-agent")

        with patch("crewai.mcp.security._mcp_secure_available", return_value=False):
            manager = MCPSecurityManager(config)
            assert manager.sign_tool({"name": "x"}) is None


# ---------------------------------------------------------------------------
# Config models accept security field
# ---------------------------------------------------------------------------


class TestConfigSecurityField:
    """Verify all MCP config models accept the optional security field."""

    def test_stdio_config_with_security(self):
        security = MCPSecurityConfig(agent_name="stdio-agent")
        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            security=security,
        )
        assert config.security is not None
        assert config.security.agent_name == "stdio-agent"

    def test_http_config_with_security(self):
        security = MCPSecurityConfig(agent_name="http-agent")
        config = MCPServerHTTP(
            url="https://example.com/mcp",
            security=security,
        )
        assert config.security is not None
        assert config.security.agent_name == "http-agent"

    def test_sse_config_with_security(self):
        security = MCPSecurityConfig(agent_name="sse-agent")
        config = MCPServerSSE(
            url="https://example.com/mcp/sse",
            security=security,
        )
        assert config.security is not None
        assert config.security.agent_name == "sse-agent"

    def test_config_without_security(self):
        """Security is optional -- configs without it should still work."""
        config = MCPServerHTTP(url="https://example.com/mcp")
        assert config.security is None


# ---------------------------------------------------------------------------
# MCPClient integration
# ---------------------------------------------------------------------------


class TestMCPClientSecurityIntegration:
    """Verify MCPClient passes signed envelopes when security_manager is set."""

    @pytest.mark.asyncio
    async def test_call_tool_signs_message(self):
        """call_tool should attach _mcps_envelope when security is enabled."""
        from crewai.mcp.client import MCPClient

        config = MCPSecurityConfig(agent_name="test-agent")
        manager = MCPSecurityManager(config)

        transport = MagicMock()
        transport.connected = True
        transport.transport_type = MagicMock()
        transport.transport_type.value = "http"
        transport.url = "https://example.com/mcp"

        client = MCPClient(transport=transport, security_manager=manager)
        client._initialized = True

        # Mock session.call_tool
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock(text="success")]
        client._session = MagicMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("read_file", {"path": "/tmp/test"})

        # Verify call_tool was called with arguments containing _mcps_envelope
        call_args = client._session.call_tool.call_args
        actual_args = call_args[0][1]  # second positional arg = arguments dict
        assert "_mcps_envelope" in actual_args
        envelope = actual_args["_mcps_envelope"]
        assert "signature" in envelope
        assert "nonce" in envelope
        assert "passport_id" in envelope

    @pytest.mark.asyncio
    async def test_call_tool_without_security(self):
        """call_tool should NOT attach _mcps_envelope when security is None."""
        from crewai.mcp.client import MCPClient

        transport = MagicMock()
        transport.connected = True
        transport.transport_type = MagicMock()
        transport.transport_type.value = "http"
        transport.url = "https://example.com/mcp"

        client = MCPClient(transport=transport, security_manager=None)
        client._initialized = True

        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock(text="success")]
        client._session = MagicMock()
        client._session.call_tool = AsyncMock(return_value=mock_result)

        await client.call_tool("read_file", {"path": "/tmp/test"})

        call_args = client._session.call_tool.call_args
        actual_args = call_args[0][1]
        assert "_mcps_envelope" not in actual_args


# ---------------------------------------------------------------------------
# MCPToolResolver integration
# ---------------------------------------------------------------------------


class TestToolResolverSecurityIntegration:
    """Verify MCPToolResolver creates a security manager from config."""

    def test_resolver_creates_security_manager(self):
        """When config has security, resolver should pass security_manager to MCPClient."""
        security = MCPSecurityConfig(agent_name="test-agent")
        http_config = MCPServerHTTP(
            url="https://example.com/mcp",
            security=security,
        )

        mock_tool_definitions = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {"type": "object"},
            }
        ]

        with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            from crewai.agent.core import Agent

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                mcps=[http_config],
            )
            tools = agent.get_mcp_tools([http_config])

            # The MCPClient constructor should have received a security_manager
            call_kwargs = mock_client_class.call_args.kwargs
            assert "security_manager" in call_kwargs
            assert call_kwargs["security_manager"] is not None
            assert isinstance(call_kwargs["security_manager"], MCPSecurityManager)

    def test_resolver_no_security_when_not_configured(self):
        """When config has no security, security_manager should be None."""
        http_config = MCPServerHTTP(url="https://example.com/mcp")

        mock_tool_definitions = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {"type": "object"},
            }
        ]

        with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
            mock_client.connected = False
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            mock_client_class.return_value = mock_client

            from crewai.agent.core import Agent

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                mcps=[http_config],
            )
            agent.get_mcp_tools([http_config])

            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("security_manager") is None

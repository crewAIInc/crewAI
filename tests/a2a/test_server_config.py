"""Tests for ServerConfig dataclass."""

import pytest

try:
    from crewai.a2a.server import ServerConfig
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestServerConfig:
    """Test ServerConfig dataclass functionality."""
    
    def test_server_config_defaults(self):
        """Test ServerConfig with default values."""
        config = ServerConfig()
        
        assert config.host == "localhost"
        assert config.port == 10001
        assert config.transport == "starlette"
        assert config.agent_name is None
        assert config.agent_description is None
    
    def test_server_config_custom_values(self):
        """Test ServerConfig with custom values."""
        config = ServerConfig(
            host="0.0.0.0",
            port=8080,
            transport="custom",
            agent_name="Test Agent",
            agent_description="A test agent"
        )
        
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.transport == "custom"
        assert config.agent_name == "Test Agent"
        assert config.agent_description == "A test agent"
    
    def test_server_config_partial_override(self):
        """Test ServerConfig with partial value override."""
        config = ServerConfig(
            port=9000,
            agent_name="Custom Agent"
        )
        
        assert config.host == "localhost"  # default
        assert config.port == 9000  # custom
        assert config.transport == "starlette"  # default
        assert config.agent_name == "Custom Agent"  # custom
        assert config.agent_description is None  # default

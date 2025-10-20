"""Shared fixtures for MCP testing."""

import pytest
from unittest.mock import Mock

# Import from the source directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from crewai.agent import Agent
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper
from tests.mocks.mcp_server_mock import MockMCPServerFactory


@pytest.fixture
def sample_mcp_agent():
    """Create a sample agent with MCP configuration for testing."""
    return Agent(
        role="Test MCP Agent",
        goal="Test MCP functionality",
        backstory="Agent designed for MCP testing",
        mcps=["https://api.test.com/mcp"]
    )


@pytest.fixture
def multi_mcp_agent():
    """Create agent with multiple MCP configurations."""
    return Agent(
        role="Multi-MCP Agent",
        goal="Test multiple MCP server integration",
        backstory="Agent with access to multiple MCP servers",
        mcps=[
            "https://search.server.com/mcp",
            "https://analysis.server.com/mcp#specific_tool",
            "crewai-amp:research-tools",
            "crewai-amp:financial-data#stock_prices"
        ]
    )


@pytest.fixture
def mcp_agent_no_tools():
    """Create agent without MCP configuration."""
    return Agent(
        role="No MCP Agent",
        goal="Test without MCP tools",
        backstory="Standard agent without MCP access"
    )


@pytest.fixture
def sample_mcp_tool_wrapper():
    """Create a sample MCPToolWrapper for testing."""
    return MCPToolWrapper(
        mcp_server_params={"url": "https://test.server.com/mcp"},
        tool_name="test_tool",
        tool_schema={
            "description": "Test tool for MCP integration",
            "args_schema": None
        },
        server_name="test_server_com_mcp"
    )


@pytest.fixture
def mock_mcp_tool_schemas():
    """Provide mock MCP tool schemas for testing."""
    return {
        "search_web": {
            "description": "Search the web for information",
            "args_schema": None
        },
        "analyze_data": {
            "description": "Analyze provided data and generate insights",
            "args_schema": None
        },
        "get_weather": {
            "description": "Get weather information for a location",
            "args_schema": None
        }
    }


@pytest.fixture
def mock_exa_like_tools():
    """Provide mock tools similar to Exa MCP server."""
    tools = []

    # Web search tool
    web_search = Mock(spec=MCPToolWrapper)
    web_search.name = "mcp_exa_ai_mcp_web_search_exa"
    web_search.description = "Search the web using Exa AI"
    web_search.original_tool_name = "web_search_exa"
    web_search.server_name = "mcp_exa_ai_mcp"
    tools.append(web_search)

    # Code context tool
    code_context = Mock(spec=MCPToolWrapper)
    code_context.name = "mcp_exa_ai_mcp_get_code_context_exa"
    code_context.description = "Get code context using Exa"
    code_context.original_tool_name = "get_code_context_exa"
    code_context.server_name = "mcp_exa_ai_mcp"
    tools.append(code_context)

    return tools


@pytest.fixture
def mock_weather_like_tools():
    """Provide mock tools similar to weather MCP server."""
    tools = []

    weather_tools = [
        ("get_current_weather", "Get current weather conditions"),
        ("get_forecast", "Get weather forecast for next 5 days"),
        ("get_alerts", "Get active weather alerts")
    ]

    for tool_name, description in weather_tools:
        tool = Mock(spec=MCPToolWrapper)
        tool.name = f"weather_server_com_mcp_{tool_name}"
        tool.description = description
        tool.original_tool_name = tool_name
        tool.server_name = "weather_server_com_mcp"
        tools.append(tool)

    return tools


@pytest.fixture
def mock_amp_mcp_responses():
    """Provide mock responses for CrewAI AMP MCP API calls."""
    return {
        "research-tools": [
            {"url": "https://amp.crewai.com/mcp/research/v1"},
            {"url": "https://amp.crewai.com/mcp/research/v2"}
        ],
        "financial-data": [
            {"url": "https://amp.crewai.com/mcp/financial/main"}
        ],
        "weather-service": [
            {"url": "https://amp.crewai.com/mcp/weather/api"}
        ]
    }


@pytest.fixture
def performance_test_mcps():
    """Provide MCP configurations for performance testing."""
    return [
        "https://fast-server.com/mcp",
        "https://medium-server.com/mcp",
        "https://reliable-server.com/mcp"
    ]


@pytest.fixture
def error_scenario_mcps():
    """Provide MCP configurations for error scenario testing."""
    return [
        "https://timeout-server.com/mcp",
        "https://auth-fail-server.com/mcp",
        "https://json-error-server.com/mcp",
        "https://not-found-server.com/mcp"
    ]


@pytest.fixture
def mixed_quality_mcps():
    """Provide mixed quality MCP server configurations for resilience testing."""
    return [
        "https://excellent-server.com/mcp",        # Always works
        "https://intermittent-server.com/mcp",     # Sometimes works
        "https://slow-but-working-server.com/mcp", # Slow but reliable
        "https://completely-broken-server.com/mcp" # Never works
    ]


@pytest.fixture
def server_name_test_cases():
    """Provide test cases for server name extraction."""
    return [
        # (input_url, expected_server_name)
        ("https://api.example.com/mcp", "api_example_com_mcp"),
        ("https://mcp.exa.ai/api/v1", "mcp_exa_ai_api_v1"),
        ("https://simple.com", "simple_com"),
        ("https://complex-domain.co.uk/deep/path/mcp", "complex-domain_co_uk_deep_path_mcp"),
        ("https://localhost:8080/mcp", "localhost:8080_mcp"),
    ]


@pytest.fixture
def mcp_reference_parsing_cases():
    """Provide test cases for MCP reference parsing."""
    return [
        # (mcp_ref, expected_type, expected_server, expected_tool)
        ("https://api.example.com/mcp", "external", "https://api.example.com/mcp", None),
        ("https://api.example.com/mcp#search", "external", "https://api.example.com/mcp", "search"),
        ("crewai-amp:weather-service", "amp", "weather-service", None),
        ("crewai-amp:financial-data#stock_price", "amp", "financial-data", "stock_price"),
    ]


@pytest.fixture
def cache_test_scenarios():
    """Provide scenarios for cache testing."""
    return {
        "cache_hit": {
            "initial_time": 1000,
            "subsequent_time": 1100,  # Within 300s TTL
            "expected_calls": 1
        },
        "cache_miss": {
            "initial_time": 1000,
            "subsequent_time": 1400,  # Beyond 300s TTL
            "expected_calls": 2
        },
        "cache_boundary": {
            "initial_time": 1000,
            "subsequent_time": 1300,  # Exactly at 300s TTL boundary
            "expected_calls": 2
        }
    }


@pytest.fixture
def timeout_test_scenarios():
    """Provide scenarios for timeout testing."""
    return {
        "connection_timeout": {
            "timeout_type": "connection",
            "delay": 15,  # Exceeds 10s connection timeout
            "expected_error": "timed out"
        },
        "execution_timeout": {
            "timeout_type": "execution",
            "delay": 35,  # Exceeds 30s execution timeout
            "expected_error": "timed out"
        },
        "discovery_timeout": {
            "timeout_type": "discovery",
            "delay": 20,  # Exceeds 15s discovery timeout
            "expected_error": "timed out"
        }
    }


@pytest.fixture
def mcp_error_scenarios():
    """Provide various MCP error scenarios for testing."""
    return {
        "connection_refused": {
            "error": ConnectionRefusedError("Connection refused"),
            "expected_msg": "network connection failed",
            "retryable": True
        },
        "auth_failed": {
            "error": Exception("Authentication failed"),
            "expected_msg": "authentication failed",
            "retryable": False
        },
        "json_parse_error": {
            "error": ValueError("JSON decode error"),
            "expected_msg": "server response parsing error",
            "retryable": True
        },
        "tool_not_found": {
            "error": Exception("Tool not found"),
            "expected_msg": "not found",
            "retryable": False
        },
        "server_error": {
            "error": Exception("Internal server error"),
            "expected_msg": "mcp execution error",
            "retryable": False
        }
    }


@pytest.fixture(autouse=True)
def clear_mcp_cache():
    """Automatically clear MCP cache before each test."""
    from crewai.agent import _mcp_schema_cache
    _mcp_schema_cache.clear()
    yield
    _mcp_schema_cache.clear()


@pytest.fixture
def mock_successful_mcp_execution():
    """Provide a mock for successful MCP tool execution."""
    def _mock_execution(**kwargs):
        return f"Successful MCP execution with args: {kwargs}"
    return _mock_execution


@pytest.fixture
def performance_benchmarks():
    """Provide performance benchmarks for MCP operations."""
    return {
        "agent_creation_max_time": 0.5,      # 500ms
        "tool_discovery_max_time": 2.0,      # 2 seconds
        "cache_hit_max_time": 0.01,          # 10ms
        "tool_execution_max_time": 35.0,     # 35 seconds (includes timeout buffer)
        "crew_integration_max_time": 0.1     # 100ms
    }


# Convenience functions for common test setup

def setup_successful_mcp_environment():
    """Set up a complete successful MCP test environment."""
    mock_server = MockMCPServerFactory.create_exa_like_server("https://mock-exa.com/mcp")

    agent = Agent(
        role="Success Test Agent",
        goal="Test successful MCP operations",
        backstory="Agent for testing successful scenarios",
        mcps=["https://mock-exa.com/mcp"]
    )

    return agent, mock_server


def setup_error_prone_mcp_environment():
    """Set up an MCP test environment with various error conditions."""
    agents = {}

    # Different agents for different error scenarios
    agents["timeout"] = Agent(
        role="Timeout Agent",
        goal="Test timeout scenarios",
        backstory="Agent for timeout testing",
        mcps=["https://slow-server.com/mcp"]
    )

    agents["auth_fail"] = Agent(
        role="Auth Fail Agent",
        goal="Test auth failures",
        backstory="Agent for auth testing",
        mcps=["https://secure-server.com/mcp"]
    )

    agents["mixed"] = Agent(
        role="Mixed Results Agent",
        goal="Test mixed success/failure",
        backstory="Agent for mixed scenario testing",
        mcps=[
            "https://working-server.com/mcp",
            "https://failing-server.com/mcp",
            "crewai-amp:working-service",
            "crewai-amp:failing-service"
        ]
    )

    return agents


def create_test_crew_with_mcp_agents(agents, task_descriptions=None):
    """Create a test crew with MCP-enabled agents."""
    if task_descriptions is None:
        task_descriptions = ["Generic test task" for _ in agents]

    tasks = []
    for i, agent in enumerate(agents):
        task = Task(
            description=task_descriptions[i] if i < len(task_descriptions) else f"Task for {agent.role}",
            expected_output=f"Output from {agent.role}",
            agent=agent
        )
        tasks.append(task)

    return Crew(agents=agents, tasks=tasks)

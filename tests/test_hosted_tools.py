import pytest
from crewai import Agent
from crewai.tools import BaseTool


class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


def test_agent_with_crewai_tools_only():
    """Test backward compatibility with CrewAI tools only."""
    mock_tool = MockTool()
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[mock_tool]
    )
    
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], BaseTool)


def test_agent_with_raw_tools_only():
    """Test agent with raw tool definitions only."""
    raw_tool = {
        "name": "hosted_search",
        "description": "Search the web using hosted search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal", 
        backstory="Test backstory",
        tools=[raw_tool]
    )
    
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], dict)
    assert agent.tools[0]["name"] == "hosted_search"


def test_agent_with_mixed_tools():
    """Test agent with both CrewAI tools and raw tool definitions."""
    mock_tool = MockTool()
    raw_tool = {
        "name": "hosted_calculator",
        "description": "Hosted calculator tool",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            }
        }
    }
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory", 
        tools=[mock_tool, raw_tool]
    )
    
    assert len(agent.tools) == 2
    assert isinstance(agent.tools[0], BaseTool)
    assert isinstance(agent.tools[1], dict)


def test_invalid_raw_tool_definition():
    """Test error handling for invalid raw tool definitions."""
    invalid_tool = {"description": "Missing name field"}
    
    with pytest.raises(ValueError, match="Raw tool definition must have a 'name' field"):
        Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            tools=[invalid_tool]
        )


def test_parse_tools_with_mixed_types():
    """Test parse_tools function with mixed tool types."""
    from crewai.utilities.agent_utils import parse_tools
    
    mock_tool = MockTool()
    raw_tool = {
        "name": "hosted_tool",
        "description": "A hosted tool",
        "parameters": {"type": "object"}
    }
    
    parsed = parse_tools([mock_tool, raw_tool])
    
    assert len(parsed) == 2
    assert hasattr(parsed[0], 'name')
    assert parsed[0].name == "mock_tool"
    assert isinstance(parsed[1], dict)
    assert parsed[1]["name"] == "hosted_tool"


def test_get_tool_names_with_mixed_types():
    """Test get_tool_names function with mixed tool types."""
    from crewai.utilities.agent_utils import get_tool_names
    
    mock_tool = MockTool()
    raw_tool = {"name": "hosted_tool", "description": "A hosted tool"}
    
    names = get_tool_names([mock_tool, raw_tool])
    assert "mock_tool" in names
    assert "hosted_tool" in names


def test_render_text_description_with_mixed_types():
    """Test render_text_description_and_args function with mixed tool types."""
    from crewai.utilities.agent_utils import render_text_description_and_args
    
    mock_tool = MockTool()
    raw_tool = {"name": "hosted_tool", "description": "A hosted tool"}
    
    description = render_text_description_and_args([mock_tool, raw_tool])
    assert "A mock tool for testing" in description
    assert "A hosted tool" in description


def test_agent_executor_with_mixed_tools():
    """Test CrewAgentExecutor initialization with mixed tool types."""
    mock_tool = MockTool()
    raw_tool = {"name": "hosted_tool", "description": "A hosted tool"}
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[mock_tool, raw_tool]
    )
    
    agent.create_agent_executor()
    
    assert len(agent.agent_executor.tool_name_to_tool_map) == 2
    assert "mock_tool" in agent.agent_executor.tool_name_to_tool_map
    assert "hosted_tool" in agent.agent_executor.tool_name_to_tool_map


def test_empty_tools_list():
    """Test agent with empty tools list."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[]
    )
    
    assert len(agent.tools) == 0


def test_none_tools():
    """Test agent with None tools."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=None
    )
    
    assert agent.tools == []


def test_raw_tool_without_description():
    """Test raw tool definition without description field."""
    raw_tool = {"name": "minimal_tool"}
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[raw_tool]
    )
    
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], dict)
    assert agent.tools[0]["name"] == "minimal_tool"


def test_complex_raw_tool_definition():
    """Test complex raw tool definition with nested parameters."""
    raw_tool = {
        "name": "complex_search",
        "description": "Advanced search with multiple parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "date_range": {"type": "string"},
                        "category": {"type": "string"}
                    }
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "required": ["query"]
        }
    }
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[raw_tool]
    )
    
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], dict)
    assert agent.tools[0]["name"] == "complex_search"
    assert "parameters" in agent.tools[0]
    assert agent.tools[0]["parameters"]["type"] == "object"

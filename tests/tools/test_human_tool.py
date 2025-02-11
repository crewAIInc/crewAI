"""Test HumanTool functionality."""

from unittest.mock import patch
import pytest

from crewai.tools import HumanTool

def test_human_tool_basic():
    """Test basic HumanTool creation and attributes."""
    tool = HumanTool()
    assert tool.name == "human"
    assert "ask user to enter input" in tool.description.lower()
    assert not tool.result_as_answer

@pytest.mark.vcr(filter_headers=["authorization"])
def test_human_tool_with_langgraph_interrupt():
    """Test HumanTool with LangGraph interrupt handling."""
    tool = HumanTool()
    
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.return_value = {"data": "test response"}
        result = tool._run("test query")
        assert result == "test response"
        mock_interrupt.assert_called_with({"query": "test query", "timeout": None})


def test_human_tool_timeout():
    """Test HumanTool timeout handling."""
    tool = HumanTool()
    timeout = 30.0
    
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.return_value = {"data": "test response"}
        result = tool._run("test query", timeout=timeout)
        assert result == "test response"
        mock_interrupt.assert_called_with({"query": "test query", "timeout": timeout})


def test_human_tool_invalid_input():
    """Test HumanTool input validation."""
    tool = HumanTool()
    
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        tool._run("")
    
    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        tool._run(None)


@pytest.mark.asyncio
async def test_human_tool_async():
    """Test async HumanTool functionality."""
    tool = HumanTool()
    
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.return_value = {"data": "test response"}
        result = await tool._arun("test query")
        assert result == "test response"
        mock_interrupt.assert_called_with({"query": "test query", "timeout": None})


@pytest.mark.asyncio
async def test_human_tool_async_timeout():
    """Test async HumanTool timeout handling."""
    tool = HumanTool()
    timeout = 30.0
    
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.return_value = {"data": "test response"}
        result = await tool._arun("test query", timeout=timeout)
        assert result == "test response"
        mock_interrupt.assert_called_with({"query": "test query", "timeout": timeout})


def test_human_tool_without_langgraph():
    """Test HumanTool behavior when LangGraph is not installed."""
    tool = HumanTool()
    
    with patch.dict('sys.modules', {'langgraph': None}):
        with pytest.raises(ImportError) as exc_info:
            tool._run("test query")
        assert "LangGraph is required" in str(exc_info.value)
        assert "pip install langgraph" in str(exc_info.value)

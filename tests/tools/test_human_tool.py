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
    
    # Test successful interrupt handling
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.return_value = {"data": "test response"}
        result = tool._run("test query")
        assert result == "test response"
        mock_interrupt.assert_called_with({"query": "test query"})

    # Test interrupt propagation
    with patch('langgraph.prebuilt.state_graphs.interrupt') as mock_interrupt:
        mock_interrupt.side_effect = Exception("Interrupt")
        with pytest.raises(Exception) as exc_info:
            tool._run("test query")
        assert "Interrupt" in str(exc_info.value)

def test_human_tool_without_langgraph():
    """Test HumanTool behavior when LangGraph is not installed."""
    tool = HumanTool()
    
    with patch.dict('sys.modules', {'langgraph': None}):
        with pytest.raises(ImportError) as exc_info:
            tool._run("test query")
        assert "LangGraph is required" in str(exc_info.value)
        assert "pip install langgraph" in str(exc_info.value)

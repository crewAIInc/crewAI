from unittest.mock import MagicMock, patch

import pytest

from crewai.tools.tool_usage import ToolUsage


class TestToolInputValidation:
    def setup_method(self):
        # Create mock objects for testing
        self.mock_tools_handler = MagicMock()
        self.mock_tools = [MagicMock()]
        self.mock_original_tools = [MagicMock()]
        self.mock_tools_description = "Mock tools description"
        self.mock_tools_names = "Mock tools names"
        self.mock_task = MagicMock()
        self.mock_function_calling_llm = MagicMock()
        
        # Create mock agent with required string attributes
        self.mock_agent = MagicMock()
        self.mock_agent.key = "mock_agent_key"
        self.mock_agent.role = "mock_agent_role"
        self.mock_agent._original_role = "mock_original_role"
        
        # Create mock action with required string attributes
        self.mock_action = MagicMock()
        self.mock_action.tool = "mock_tool_name"
        self.mock_action.tool_input = "mock_tool_input"
        
        # Create ToolUsage instance
        self.tool_usage = ToolUsage(
            tools_handler=self.mock_tools_handler,
            tools=self.mock_tools,
            original_tools=self.mock_original_tools,
            tools_description=self.mock_tools_description,
            tools_names=self.mock_tools_names,
            task=self.mock_task,
            function_calling_llm=self.mock_function_calling_llm,
            agent=self.mock_agent,
            action=self.mock_action,
        )
        
        # Patch the _emit_validate_input_error method to avoid event emission
        self.original_emit_validate_input_error = self.tool_usage._emit_validate_input_error
        self.tool_usage._emit_validate_input_error = MagicMock()

    def teardown_method(self):
        # Restore the original method
        if hasattr(self, 'original_emit_validate_input_error'):
            self.tool_usage._emit_validate_input_error = self.original_emit_validate_input_error

    def test_validate_tool_input_with_dict(self):
        # Test with a valid dictionary input
        tool_input = '{"ticker": "VST"}'
        result = self.tool_usage._validate_tool_input(tool_input)
        assert result == {"ticker": "VST"}

    def test_validate_tool_input_with_list(self):
        # Test with a list input containing a dictionary as the first element
        tool_input = '[{"ticker": "VST"}, {"tool_code": "Stock Info", "tool_input": {"ticker": "VST"}}]'
        result = self.tool_usage._validate_tool_input(tool_input)
        assert result == {"ticker": "VST"}

    def test_validate_tool_input_with_empty_list(self):
        # Test with an empty list input
        tool_input = '[]'
        with pytest.raises(Exception) as excinfo:
            self.tool_usage._validate_tool_input(tool_input)
        assert "Tool input must be a valid dictionary in JSON or Python literal format" in str(excinfo.value)

    def test_validate_tool_input_with_list_of_non_dicts(self):
        # Test with a list input containing non-dictionary elements
        tool_input = '["not a dict", 123]'
        with pytest.raises(Exception) as excinfo:
            self.tool_usage._validate_tool_input(tool_input)
        assert "Tool input must be a valid dictionary in JSON or Python literal format" in str(excinfo.value)

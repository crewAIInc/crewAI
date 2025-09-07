import json
from unittest.mock import patch

from crewai.agents.parser import AgentAction, AgentFinish
from crewai.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterJSON:
    """Test ConsoleFormatter JSON formatting functionality."""

    def test_handle_agent_logs_execution_with_json_tool_input(self):
        """Test that JSON tool inputs are properly formatted."""
        formatter = ConsoleFormatter()

        json_input = (
            '{"task": "Research AI", "context": "Machine Learning", '
            '"priority": "high"}'
        )
        agent_action = AgentAction(
            thought="I need to research this topic",
            tool="research_tool",
            tool_input=json_input,
            text="Full agent text",
            result="Research completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_handle_agent_logs_execution_with_malformed_json(self):
        """Test that malformed JSON falls back to string formatting."""
        formatter = ConsoleFormatter()

        malformed_json = (
            '{"task": "Research AI", "context": "Machine Learning"'
        )
        agent_action = AgentAction(
            thought="I need to research this topic",
            tool="research_tool",
            tool_input=malformed_json,
            text="Full agent text",
            result="Research completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_handle_agent_logs_execution_with_non_json_string(self):
        """Test that non-JSON strings are handled properly."""
        formatter = ConsoleFormatter()

        plain_string = "search for weather in San Francisco"
        agent_action = AgentAction(
            thought="I need to search for weather",
            tool="search_tool",
            tool_input=plain_string,
            text="Full agent text",
            result="Weather found"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_handle_agent_logs_execution_with_complex_json(self):
        """Test with complex nested JSON structures."""
        formatter = ConsoleFormatter()

        complex_json = json.dumps({
            "query": {
                "type": "research",
                "parameters": {
                    "topic": "AI in healthcare",
                    "depth": "comprehensive",
                    "sources": ["academic", "industry", "news"]
                }
            },
            "filters": ["recent", "peer-reviewed"],
            "limit": 50
        })

        agent_action = AgentAction(
            thought="Complex research query",
            tool="advanced_search",
            tool_input=complex_json,
            text="Full agent text",
            result="Complex search completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_handle_agent_logs_execution_with_agent_finish(self):
        """Test that AgentFinish objects are handled correctly."""
        formatter = ConsoleFormatter()

        agent_finish = AgentFinish(
            thought="Task completed",
            output="Final result of the task",
            text="Full agent text"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_finish, "Test Agent")

            mock_print.assert_called()

    def test_json_parsing_preserves_structure(self):
        """Test that JSON parsing preserves the original structure."""
        formatter = ConsoleFormatter()

        original_data = {
            "nested": {
                "array": [1, 2, 3],
                "string": "test",
                "boolean": True,
                "null": None
            }
        }
        json_string = json.dumps(original_data)

        agent_action = AgentAction(
            thought="Testing structure preservation",
            tool="test_tool",
            tool_input=json_string,
            text="Full agent text",
            result="Test completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_empty_tool_input_handling(self):
        """Test handling of empty tool input."""
        formatter = ConsoleFormatter()

        agent_action = AgentAction(
            thought="Empty input test",
            tool="test_tool",
            tool_input="",
            text="Full agent text",
            result="Test completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

    def test_numeric_tool_input_handling(self):
        """Test handling of numeric tool input."""
        formatter = ConsoleFormatter()

        agent_action = AgentAction(
            thought="Numeric input test",
            tool="test_tool",
            tool_input="42",
            text="Full agent text",
            result="Test completed"
        )

        with patch.object(formatter, 'print') as mock_print:
            formatter.handle_agent_logs_execution(agent_action, "Test Agent")

            mock_print.assert_called()

import pytest

from crewai.agents import parser
from crewai.agents.crew_agent_executor import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)


def test_valid_action_parsing_special_characters():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what's the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_json_tool_input():
    text = """
    Thought: Let's find the information
    Action: query
    Action Input: ** {"task": "What are some common challenges or barriers that you have observed or experienced when implementing AI-powered solutions in healthcare settings?", "context": "As we've discussed recent advancements in AI applications in healthcare, it's crucial to acknowledge the potential hurdles. Some possible obstacles include...", "coworker": "Senior Researcher"}
    """
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    expected_tool_input = '{"task": "What are some common challenges or barriers that you have observed or experienced when implementing AI-powered solutions in healthcare settings?", "context": "As we\'ve discussed recent advancements in AI applications in healthcare, it\'s crucial to acknowledge the potential hurdles. Some possible obstacles include...", "coworker": "Senior Researcher"}'
    assert result.tool == "query"
    assert result.tool_input == expected_tool_input


def test_valid_action_parsing_with_quotes():
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "temperature in SF"


def test_valid_action_parsing_with_curly_braces():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: {temperature in SF}"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "{temperature in SF}"


def test_valid_action_parsing_with_angle_brackets():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: <temperature in SF>"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "<temperature in SF>"


def test_valid_action_parsing_with_parentheses():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: (temperature in SF)"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "(temperature in SF)"


def test_valid_action_parsing_with_mixed_brackets():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: [temperature in {SF}]"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[temperature in {SF}]"


def test_valid_action_parsing_with_nested_quotes():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in 'SF'?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in 'SF'?"


def test_valid_action_parsing_with_incomplete_json():
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_special_characters():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? @$%^&*"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? @$%^&*"


def test_valid_action_parsing_with_combination():
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "[what is the temperature in SF?]"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[what is the temperature in SF?]"


def test_valid_action_parsing_with_mixed_quotes():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in SF?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_newlines():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is\nthe temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is\nthe temperature in SF?"


def test_valid_action_parsing_with_escaped_characters():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? \\n"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? \\n"


def test_valid_action_parsing_with_json_string():
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"}'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_unbalanced_quotes():
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what is the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_clean_action_no_formatting():
    action = "Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_asterisks():
    action = "** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_trailing_asterisks():
    action = "Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_and_trailing_asterisks():
    action = "** Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_leading_asterisks():
    action = "**** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_trailing_asterisks():
    action = "Ask question to senior researcher ****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_spaces_and_asterisks():
    action = "  **  Ask question to senior researcher  **  "
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_only_asterisks():
    action = "****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_clean_action_with_empty_string():
    action = ""
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_valid_final_answer_parsing():
    text = (
        "Thought: I found the information\nFinal Answer: The temperature is 100 degrees"
    )
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.output == "The temperature is 100 degrees"


def test_missing_action_error():
    text = "Thought: Let's find the temperature\nAction Input: what is the temperature in SF?"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "Invalid Format: I missed the 'Action:' after 'Thought:'." in str(
        exc_info.value
    )


def test_missing_action_input_error():
    text = "Thought: Let's find the temperature\nAction: search"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "I missed the 'Action Input:' after 'Action:'." in str(exc_info.value)


def test_safe_repair_json():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": Senior Researcher'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unrepairable():
    invalid_json = "{invalid_json"
    result = parser._safe_repair_json(invalid_json)
    assert result == invalid_json  # Should return the original if unrepairable


def test_safe_repair_json_missing_quotes():
    invalid_json = (
        '{task: "Research XAI", context: "Explainable AI", coworker: Senior Researcher}'
    )
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unclosed_brackets():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_extra_commas():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_trailing_commas():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_single_quotes():
    invalid_json = "{'task': 'Research XAI', 'context': 'Explainable AI', 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_mixed_quotes():
    invalid_json = "{'task': \"Research XAI\", 'context': \"Explainable AI\", 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unescaped_characters():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher\n"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_colon():
    invalid_json = '{"task" "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_comma():
    invalid_json = '{"task": "Research XAI" "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unexpected_trailing_characters():
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"} random text'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_special_characters_key():
    invalid_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    expected_repaired_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_parsing_with_whitespace():
    text = " Thought: Let's find the temperature \n Action: search \n Action Input: what is the temperature in SF? "
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_parsing_with_special_characters():
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "what is the temperature in SF?"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_integration_valid_and_invalid():
    text = """
    Thought: Let's find the temperature
    Action: search
    Action Input: what is the temperature in SF?

    Thought: I found the information
    Final Answer: The temperature is 100 degrees

    Thought: Missing action
    Action Input: invalid

    Thought: Missing action input
    Action: invalid
    """
    parts = text.strip().split("\n\n")
    results = []

    def parse_part(part_text):
        try:
            return parser.parse(part_text.strip())
        except OutputParserException as e:
            return e

    for part in parts:
        result = parse_part(part)
        results.append(result)

    assert isinstance(results[0], AgentAction)
    assert isinstance(results[1], AgentFinish)
    assert isinstance(results[2], OutputParserException)
    assert isinstance(results[3], OutputParserException)


# TODO: ADD TEST TO MAKE SURE ** REMOVAL DOESN'T MESS UP ANYTHING


def test_harmony_analysis_channel_parsing():
    """Test parsing OpenAI Harmony analysis channel (final answer)."""
    text = "<|start|>assistant<|channel|>analysis<|message|>The temperature in SF is 72°F<|end|>"
    result = parser.parse(text)
    assert isinstance(result, parser.AgentFinish)
    assert result.output == "The temperature in SF is 72°F"
    assert "Analysis:" in result.thought


def test_harmony_commentary_channel_parsing():
    """Test parsing OpenAI Harmony commentary channel (tool action)."""
    text = '<|start|>assistant<|channel|>commentary to=search<|message|>{"query": "temperature in SF"}<|call|>'
    result = parser.parse(text)
    assert isinstance(result, parser.AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_harmony_commentary_with_thought():
    """Test Harmony commentary with reasoning before JSON."""
    text = '<|start|>assistant<|channel|>commentary to=search<|message|>I need to find the temperature {"query": "SF weather"}<|call|>'
    result = parser.parse(text)
    assert isinstance(result, parser.AgentAction)
    assert result.tool == "search"
    assert result.thought == "I need to find the temperature"
    assert result.tool_input == '{"query": "SF weather"}'


def test_harmony_multiple_blocks():
    """Test parsing multiple Harmony blocks (uses last one)."""
    text = '''<|start|>assistant<|channel|>analysis<|message|>Thinking about this<|end|>
<|start|>assistant<|channel|>commentary to=search<|message|>{"query": "test"}<|call|>'''
    result = parser.parse(text)
    assert isinstance(result, parser.AgentAction)
    assert result.tool == "search"


def test_harmony_format_detection():
    """Test that Harmony format is properly detected."""
    harmony_text = "<|start|>assistant<|channel|>analysis<|message|>result<|end|>"
    react_text = "Thought: test\nFinal Answer: result"

    harmony_result = parser.parse(harmony_text)
    react_result = parser.parse(react_text)

    assert isinstance(harmony_result, parser.AgentFinish)
    assert isinstance(react_result, parser.AgentFinish)
    assert harmony_result.output == "result"
    assert react_result.output == "result"


def test_harmony_invalid_format_error():
    """Test error handling for invalid Harmony format."""
    text = "<|start|>assistant<|channel|>unknown<|message|>content<|end|>"
    with pytest.raises(parser.OutputParserException) as exc_info:
        parser.parse(text)
    assert "Invalid Harmony Format" in str(exc_info.value)


def test_automatic_format_detection():
    """Test that the parser automatically detects different formats."""
    react_action = "Thought: Let's search\nAction: search\nAction Input: query"
    react_finish = "Thought: Done\nFinal Answer: result"

    harmony_action = '<|start|>assistant<|channel|>commentary to=tool<|message|>{"input": "test"}<|call|>'
    harmony_finish = "<|start|>assistant<|channel|>analysis<|message|>final result<|end|>"

    assert isinstance(parser.parse(react_action), parser.AgentAction)
    assert isinstance(parser.parse(react_finish), parser.AgentFinish)
    assert isinstance(parser.parse(harmony_action), parser.AgentAction)
    assert isinstance(parser.parse(harmony_finish), parser.AgentFinish)


def test_format_registry():
    """Test the format registry functionality."""
    from crewai.agents.parser import _format_registry

    assert 'react' in _format_registry._parsers
    assert 'harmony' in _format_registry._parsers

    react_text = "Thought: test\nAction: search\nAction Input: query"
    harmony_text = "<|start|>assistant<|channel|>analysis<|message|>result<|end|>"

    assert _format_registry._parsers['react'].can_parse(react_text)
    assert _format_registry._parsers['harmony'].can_parse(harmony_text)
    assert not _format_registry._parsers['react'].can_parse(harmony_text)
    assert not _format_registry._parsers['harmony'].can_parse(react_text)


def test_backward_compatibility():
    """Test that all existing ReAct format tests still pass."""

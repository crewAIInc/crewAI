import pytest
from crewai.agents import parser
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
)
from crewai.agents.parser import (
    OutputParserError as OutputParserException,
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
    for part in parts:
        try:
            result = parser.parse(part.strip())
        except OutputParserException as e:
            result = e
        results.append(result)

    assert isinstance(results[0], AgentAction)
    assert isinstance(results[1], AgentFinish)
    assert isinstance(results[2], OutputParserException)
    assert isinstance(results[3], OutputParserException)


# TODO: ADD TEST TO MAKE SURE ** REMOVAL DOESN'T MESS UP ANYTHING


# Tests for Action: None handling (Issue #4186)
def test_action_none_basic():
    """Test that 'Action: None' is parsed as AgentFinish."""
    text = "Thought: I cannot use any tool for this.\nAction: None"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "I cannot use any tool for this." in result.output


def test_action_none_with_reason_in_parentheses():
    """Test 'Action: None (reason)' format."""
    text = "Thought: The tool is not available.\nAction: None (direct response required)"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "The tool is not available." in result.output


def test_action_none_lowercase():
    """Test that 'Action: none' (lowercase) is handled."""
    text = "Thought: I should respond directly.\nAction: none"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "I should respond directly." in result.output


def test_action_na():
    """Test that 'Action: N/A' is handled."""
    text = "Thought: No action needed here.\nAction: N/A"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "No action needed here." in result.output


def test_action_na_lowercase():
    """Test that 'Action: n/a' (lowercase) is handled."""
    text = "Thought: This requires a direct answer.\nAction: n/a"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "This requires a direct answer." in result.output


def test_action_none_with_dash_separator():
    """Test 'Action: None - reason' format."""
    text = "Thought: I need to provide a direct response.\nAction: None - direct response"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "I need to provide a direct response." in result.output


def test_action_none_with_additional_content():
    """Test 'Action: None' with additional content after."""
    text = "Thought: I analyzed the request.\nAction: None\nHere is my direct response to your question."
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "I analyzed the request." in result.output


def test_action_no_action():
    """Test that 'Action: no action' is handled."""
    text = "Thought: I will respond without using tools.\nAction: no action"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "I will respond without using tools." in result.output


def test_action_none_without_thought():
    """Test 'Action: None' without a thought prefix."""
    text = "Action: None"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.output == "I cannot perform this action with the available tools."


def test_action_none_preserves_original_text():
    """Test that the original text is preserved in the result."""
    text = "Thought: I cannot delegate this task.\nAction: None"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.text == text


def test_action_none_with_colon_separator():
    """Test 'Action: None: reason' format."""
    text = "Thought: Direct response needed.\nAction: None: providing direct answer"
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert "Direct response needed." in result.output

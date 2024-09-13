import pytest
from crewai.agents.parser import CrewAgentParser
from crewai.agents.crew_agent_executor import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)


@pytest.fixture
def parser():
    agent = MockAgent()
    p = CrewAgentParser(agent)
    return p


def test_valid_action_parsing_special_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what's the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_json_tool_input(parser):
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


def test_valid_action_parsing_with_quotes(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "temperature in SF"


def test_valid_action_parsing_with_curly_braces(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: {temperature in SF}"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "{temperature in SF}"


def test_valid_action_parsing_with_angle_brackets(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: <temperature in SF>"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "<temperature in SF>"


def test_valid_action_parsing_with_parentheses(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: (temperature in SF)"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "(temperature in SF)"


def test_valid_action_parsing_with_mixed_brackets(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: [temperature in {SF}]"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[temperature in {SF}]"


def test_valid_action_parsing_with_nested_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in 'SF'?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in 'SF'?"


def test_valid_action_parsing_with_incomplete_json(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_special_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? @$%^&*"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? @$%^&*"


def test_valid_action_parsing_with_combination(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "[what is the temperature in SF?]"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "[what is the temperature in SF?]"


def test_valid_action_parsing_with_mixed_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what's the temperature in SF?\""
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what's the temperature in SF?"


def test_valid_action_parsing_with_newlines(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is\nthe temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is\nthe temperature in SF?"


def test_valid_action_parsing_with_escaped_characters(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: what is the temperature in SF? \\n"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF? \\n"


def test_valid_action_parsing_with_json_string(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: {"query": "temperature in SF"}'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == '{"query": "temperature in SF"}'


def test_valid_action_parsing_with_unbalanced_quotes(parser):
    text = "Thought: Let's find the temperature\nAction: search\nAction Input: \"what is the temperature in SF?"
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_clean_action_no_formatting(parser):
    action = "Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_asterisks(parser):
    action = "** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_trailing_asterisks(parser):
    action = "Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_leading_and_trailing_asterisks(parser):
    action = "** Ask question to senior researcher **"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_leading_asterisks(parser):
    action = "**** Ask question to senior researcher"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_multiple_trailing_asterisks(parser):
    action = "Ask question to senior researcher ****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_spaces_and_asterisks(parser):
    action = "  **  Ask question to senior researcher  **  "
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == "Ask question to senior researcher"


def test_clean_action_with_only_asterisks(parser):
    action = "****"
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_clean_action_with_empty_string(parser):
    action = ""
    cleaned_action = parser._clean_action(action)
    assert cleaned_action == ""


def test_valid_final_answer_parsing(parser):
    text = (
        "Thought: I found the information\nFinal Answer: The temperature is 100 degrees"
    )
    result = parser.parse(text)
    assert isinstance(result, AgentFinish)
    assert result.output == "The temperature is 100 degrees"


def test_missing_action_error(parser):
    text = "Thought: Let's find the temperature\nAction Input: what is the temperature in SF?"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "Invalid Format: I missed the 'Action:' after 'Thought:'." in str(
        exc_info.value
    )


def test_missing_action_input_error(parser):
    text = "Thought: Let's find the temperature\nAction: search"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "I missed the 'Action Input:' after 'Action:'." in str(exc_info.value)


def test_action_and_final_answer_error(parser):
    text = "Thought: I found the information\nAction: search\nAction Input: what is the temperature in SF?\nFinal Answer: The temperature is 100 degrees"
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text)
    assert "both perform Action and give a Final Answer" in str(exc_info.value)


def test_safe_repair_json(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": Senior Researcher'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unrepairable(parser):
    invalid_json = "{invalid_json"
    result = parser._safe_repair_json(invalid_json)
    assert result == invalid_json  # Should return the original if unrepairable


def test_safe_repair_json_missing_quotes(parser):
    invalid_json = (
        '{task: "Research XAI", context: "Explainable AI", coworker: Senior Researcher}'
    )
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unclosed_brackets(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_extra_commas(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_trailing_commas(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher",}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_single_quotes(parser):
    invalid_json = "{'task': 'Research XAI', 'context': 'Explainable AI', 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_mixed_quotes(parser):
    invalid_json = "{'task': \"Research XAI\", 'context': \"Explainable AI\", 'coworker': 'Senior Researcher'}"
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unescaped_characters(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher\n"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_colon(parser):
    invalid_json = '{"task" "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_missing_comma(parser):
    invalid_json = '{"task": "Research XAI" "context": "Explainable AI", "coworker": "Senior Researcher"}'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_unexpected_trailing_characters(parser):
    invalid_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"} random text'
    expected_repaired_json = '{"task": "Research XAI", "context": "Explainable AI", "coworker": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_safe_repair_json_special_characters_key(parser):
    invalid_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    expected_repaired_json = '{"task!@#": "Research XAI", "context$%^": "Explainable AI", "coworker&*()": "Senior Researcher"}'
    result = parser._safe_repair_json(invalid_json)
    assert result == expected_repaired_json


def test_parsing_with_whitespace(parser):
    text = " Thought: Let's find the temperature \n Action: search \n Action Input: what is the temperature in SF? "
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_parsing_with_special_characters(parser):
    text = 'Thought: Let\'s find the temperature\nAction: search\nAction Input: "what is the temperature in SF?"'
    result = parser.parse(text)
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"


def test_integration_valid_and_invalid(parser):
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
            results.append(result)
        except OutputParserException as e:
            results.append(e)

    assert isinstance(results[0], AgentAction)
    assert isinstance(results[1], AgentFinish)
    assert isinstance(results[2], OutputParserException)
    assert isinstance(results[3], OutputParserException)


class MockAgent:
    def increment_formatting_errors(self):
        pass


# TODO: ADD TEST TO MAKE SURE ** REMOVAL DOESN'T MESS UP ANYTHING

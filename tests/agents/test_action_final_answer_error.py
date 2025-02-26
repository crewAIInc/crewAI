import pytest
from unittest.mock import MagicMock

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    OutputParserException,
)


def test_process_llm_response_with_action_and_final_answer():
    """Test that _process_llm_response correctly handles outputs with both Action and Final Answer."""
    # Create a mock LLM
    mock_llm = MagicMock()
    mock_llm.supports_stop_words.return_value = False
    
    # Create a mock agent
    mock_agent = MagicMock()
    
    # Create a CrewAgentExecutor instance
    executor = CrewAgentExecutor(
        llm=mock_llm,
        task=MagicMock(),
        crew=MagicMock(),
        agent=mock_agent,
        prompt={},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock(),
    )
    
    # Test case 1: Output with both Action and Final Answer, with Final Answer after Action
    output_with_both = """
    Thought: I need to search for information and then provide an answer.
    Action: search
    Action Input: what is the temperature in SF?
    Final Answer: The temperature is 100 degrees
    """
    
    # Mock the _format_answer method to first raise an exception and then return a valid result
    format_answer_mock = MagicMock()
    format_answer_mock.side_effect = [
        OutputParserException(FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE),
        AgentAction(thought="", tool="search", tool_input="what is the temperature in SF?", text=""),
    ]
    executor._format_answer = format_answer_mock
    
    # Process the response
    result = executor._process_llm_response(output_with_both)
    
    # Verify that the result is an AgentAction
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"
    
    # Test case 2: Output with both Action and Final Answer, with Observation in between
    output_with_observation = """
    Thought: I need to search for information.
    Action: search
    Action Input: what is the temperature in SF?
    Observation: The temperature in SF is 100 degrees.
    Final Answer: The temperature is 100 degrees
    """
    
    # Reset the mock
    format_answer_mock.reset_mock()
    format_answer_mock.side_effect = [
        OutputParserException(FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE),
        AgentAction(thought="", tool="search", tool_input="what is the temperature in SF?", text=""),
    ]
    
    # Process the response
    result = executor._process_llm_response(output_with_observation)
    
    # Verify that the result is an AgentAction
    assert isinstance(result, AgentAction)
    assert result.tool == "search"
    assert result.tool_input == "what is the temperature in SF?"

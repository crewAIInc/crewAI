"""Tests for agent output sanitization to prevent internal fields from leaking."""

import pytest
from unittest.mock import Mock, patch

from crewai import Agent, Crew, Task
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.process import Process
from crewai.utilities.agent_utils import (
    format_answer,
    handle_max_iterations_exceeded,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns ReAct-style output."""
    llm = Mock()
    llm.call = Mock()
    llm.supports_stop_words = Mock(return_value=True)
    llm.get_context_window_size = Mock(return_value=4096)
    return llm


@pytest.fixture
def mock_printer():
    """Create a mock printer."""
    printer = Mock()
    printer.print = Mock()
    return printer


@pytest.fixture
def mock_i18n():
    """Create a mock i18n."""
    i18n = Mock()
    i18n.errors = Mock(return_value="Please provide a final answer.")
    return i18n


def test_handle_max_iterations_with_agent_action_should_not_leak_internal_fields(
    mock_llm, mock_printer, mock_i18n
):
    """Test that when max iterations is exceeded and we have an AgentAction,
    the final output doesn't contain internal ReAct fields like 'Thought:' and 'Action:'.
    
    This reproduces issue #3873 where hierarchical crews would return internal
    fields in the final answer when delegated tasks failed.
    """
    formatted_answer = AgentAction(
        thought="I need to fetch the database tables",
        tool="PostgresTool",
        tool_input="list_tables",
        text="Thought: I need to fetch the database tables\nAction: PostgresTool\nAction Input: list_tables",
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Fetch list of tables from postgres db"},
    ]
    
    mock_llm.call.return_value = (
        "Thought: I should try to connect to the database\n"
        "Action: PostgresTool\n"
        "Action Input: connect"
    )
    
    callbacks = []
    
    result = handle_max_iterations_exceeded(
        formatted_answer=formatted_answer,
        printer=mock_printer,
        i18n=mock_i18n,
        messages=messages,
        llm=mock_llm,
        callbacks=callbacks,
    )
    
    assert isinstance(result, AgentFinish)
    
    assert "Thought:" not in result.output, (
        f"Output should not contain 'Thought:' but got: {result.output}"
    )
    assert "Action:" not in result.output, (
        f"Output should not contain 'Action:' but got: {result.output}"
    )
    assert "Action Input:" not in result.output, (
        f"Output should not contain 'Action Input:' but got: {result.output}"
    )


def test_format_answer_with_unparseable_output_should_not_leak_internal_fields():
    """Test that when format_answer receives unparseable output with ReAct fields,
    it sanitizes them from the final output.
    """
    raw_answer = (
        "Thought: I tried to connect to the database but failed\n"
        "Action: PostgresTool\n"
        "Action Input: connect\n"
        "Observation: Error: Database configuration not found"
    )
    
    with patch("crewai.utilities.agent_utils.parse") as mock_parse:
        mock_parse.side_effect = Exception("Failed to parse")
        
        result = format_answer(raw_answer)
    
    assert isinstance(result, AgentFinish)
    
    assert "Thought:" not in result.output, (
        f"Output should not contain 'Thought:' but got: {result.output}"
    )
    assert "Action:" not in result.output, (
        f"Output should not contain 'Action:' but got: {result.output}"
    )
    assert "Action Input:" not in result.output, (
        f"Output should not contain 'Action Input:' but got: {result.output}"
    )
    assert "Observation:" not in result.output, (
        f"Output should not contain 'Observation:' but got: {result.output}"
    )


def test_hierarchical_crew_with_failing_task_should_not_leak_internal_fields():
    """Integration test: hierarchical crew with a failing delegated task
    should not leak internal ReAct fields in the final output.
    
    This is a full integration test that reproduces issue #3873.
    
    Note: This test is skipped for now as it requires VCR cassettes.
    The unit tests above cover the core functionality.
    """
    pytest.skip("Integration test requires VCR cassettes - covered by unit tests")
    expert = Agent(
        role="Database Expert",
        goal="Fetch database information",
        backstory="You are an expert in database operations.",
        max_iter=2,  # Set low max_iter to trigger the bug
        verbose=True,
    )
    
    task = Task(
        description="Fetch list of tables from postgres database",
        expected_output="A list of database tables",
        agent=expert,
    )
    
    crew = Crew(
        agents=[expert],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        verbose=True,
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    assert "Thought:" not in result.raw, (
        f"Final output should not contain 'Thought:' but got: {result.raw}"
    )
    assert "Action:" not in result.raw, (
        f"Final output should not contain 'Action:' but got: {result.raw}"
    )
    assert "Action Input:" not in result.raw, (
        f"Final output should not contain 'Action Input:' but got: {result.raw}"
    )

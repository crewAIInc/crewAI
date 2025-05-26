"""Tests for reasoning interval and adaptive reasoning in agents."""

import pytest
from unittest.mock import patch, MagicMock

from crewai import Agent, Task
from crewai.llm import LLM
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.utilities.reasoning_handler import AgentReasoning


@pytest.fixture
def mock_llm_responses():
    """Fixture for mock LLM responses."""
    return {
        "initial_reasoning": "I'll solve this task step by step.\n\nREADY: I am ready to execute the task.\n\n",
        "mid_execution_reasoning": "Based on progress so far, I'll adjust my approach.\n\nREADY: I am ready to continue executing the task.",
        "execution_step": "I'm working on the task...",
        "final_result": "Task completed successfully."
    }


def test_agent_with_reasoning_interval(mock_llm_responses):
    """Test agent with reasoning interval."""
    with patch('crewai.llm.LLM.call') as mock_llm_call:
        mock_llm_call.return_value = mock_llm_responses["initial_reasoning"]
        
        llm = MagicMock()
        llm.call.return_value = mock_llm_responses["initial_reasoning"]
        
        agent = Agent(
            role="Test Agent",
            goal="To test the reasoning interval feature",
            backstory="I am a test agent created to verify the reasoning interval feature works correctly.",
            llm=llm,
            reasoning=True,
            reasoning_interval=2,  # Reason every 2 steps
            verbose=True
        )
    
    task = Task(
        description="Multi-step task that requires periodic reasoning.",
        expected_output="The task should be completed with periodic reasoning.",
        agent=agent
    )
    
    with patch('crewai.agent.Agent.create_agent_executor') as mock_create_executor:
        mock_executor = MagicMock()
        mock_executor._handle_mid_execution_reasoning = MagicMock()
        mock_executor.invoke.return_value = mock_llm_responses["final_result"]
        mock_create_executor.return_value = mock_executor
        
        result = agent.execute_task(task)
        
        assert result == mock_llm_responses["final_result"]
        
        mock_executor._handle_mid_execution_reasoning.assert_called()


def test_agent_with_adaptive_reasoning(mock_llm_responses):
    """Test agent with adaptive reasoning."""
    with patch('crewai.llm.LLM.call') as mock_llm_call:
        mock_llm_call.return_value = mock_llm_responses["initial_reasoning"]
        
        llm = MagicMock()
        llm.call.return_value = mock_llm_responses["initial_reasoning"]
        
        agent = Agent(
            role="Test Agent",
            goal="To test the adaptive reasoning feature",
            backstory="I am a test agent created to verify the adaptive reasoning feature works correctly.",
            llm=llm,
            reasoning=True,
            adaptive_reasoning=True,
            verbose=True
        )
    
    task = Task(
        description="Complex task that requires adaptive reasoning.",
        expected_output="The task should be completed with adaptive reasoning.",
        agent=agent
    )
    
    with patch('crewai.agent.Agent.create_agent_executor') as mock_create_executor:
        mock_executor = MagicMock()
        mock_executor._should_adaptive_reason = MagicMock(return_value=True)
        mock_executor._handle_mid_execution_reasoning = MagicMock()
        mock_executor.invoke.return_value = mock_llm_responses["final_result"]
        mock_create_executor.return_value = mock_executor
        
        result = agent.execute_task(task)
        
        assert result == mock_llm_responses["final_result"]
        
        mock_executor._should_adaptive_reason.assert_called()
        
        mock_executor._handle_mid_execution_reasoning.assert_called()


def test_mid_execution_reasoning_handler():
    """Test the mid-execution reasoning handler."""
    llm = MagicMock()
    llm.call.return_value = "Based on progress, I'll adjust my approach.\n\nREADY: I am ready to continue executing the task."
    
    agent = Agent(
        role="Test Agent",
        goal="To test the mid-execution reasoning handler",
        backstory="I am a test agent created to verify the mid-execution reasoning handler works correctly.",
        llm=llm,
        reasoning=True,
        verbose=True
    )
    
    task = Task(
        description="Task to test mid-execution reasoning handler.",
        expected_output="The mid-execution reasoning handler should work correctly.",
        agent=agent
    )
    
    agent.llm.call = MagicMock(return_value="Based on progress, I'll adjust my approach.\n\nREADY: I am ready to continue executing the task.")
    
    reasoning_handler = AgentReasoning(task=task, agent=agent)
    
    result = reasoning_handler.handle_mid_execution_reasoning(
        current_steps=3,
        tools_used=["search_tool", "calculator_tool"],
        current_progress="Made progress on steps 1-3",
        iteration_messages=[
            {"role": "assistant", "content": "I'll search for information."},
            {"role": "system", "content": "Search results: ..."},
            {"role": "assistant", "content": "I'll calculate the answer."},
            {"role": "system", "content": "Calculation result: 42"}
        ]
    )
    
    assert result is not None
    assert hasattr(result, 'plan')
    assert hasattr(result.plan, 'plan')
    assert hasattr(result.plan, 'ready')
    assert result.plan.ready is True


def test_should_trigger_reasoning_interval():
    """Test the _should_trigger_reasoning method with interval-based reasoning."""
    agent = MagicMock()
    agent.reasoning = True
    agent.reasoning_interval = 3
    agent.adaptive_reasoning = False
    
    executor = CrewAgentExecutor(
        llm=MagicMock(),
        task=MagicMock(),
        crew=MagicMock(),
        agent=agent,
        prompt={},
        max_iter=10,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock()
    )
    
    executor.steps_since_reasoning = 0
    assert executor._should_trigger_reasoning() is False
    
    executor.steps_since_reasoning = 2
    assert executor._should_trigger_reasoning() is False
    
    executor.steps_since_reasoning = 3
    assert executor._should_trigger_reasoning() is True
    
    executor.steps_since_reasoning = 4
    assert executor._should_trigger_reasoning() is True


def test_should_trigger_adaptive_reasoning():
    """Test the _should_adaptive_reason method."""
    agent = MagicMock()
    agent.reasoning = True
    agent.reasoning_interval = None
    agent.adaptive_reasoning = True
    
    executor = CrewAgentExecutor(
        llm=MagicMock(),
        task=MagicMock(),
        crew=MagicMock(),
        agent=agent,
        prompt={},
        max_iter=10,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock()
    )
    
    executor.tools_used = ["tool1", "tool2", "tool3"]
    assert executor._should_adaptive_reason() is True
    
    executor.tools_used = ["tool1", "tool1", "tool1"]
    executor.iterations = 6  # > max_iter // 2
    assert executor._should_adaptive_reason() is True
    
    executor.tools_used = ["tool1", "tool1", "tool1"]
    executor.iterations = 2
    executor.messages = [
        {"role": "assistant", "content": "I'll try this approach."},
        {"role": "system", "content": "Error: Failed to execute the command."},
        {"role": "assistant", "content": "Let me try something else."}
    ]
    assert executor._should_adaptive_reason() is True
    
    executor.tools_used = ["tool1", "tool1", "tool1"]
    executor.iterations = 2
    executor.messages = [
        {"role": "assistant", "content": "I'll try this approach."},
        {"role": "system", "content": "Command executed successfully."},
        {"role": "assistant", "content": "Let me continue with the next step."}
    ]
    assert executor._should_adaptive_reason() is False


@pytest.mark.parametrize("interval,steps,should_reason", [
    (None, 5, False),
    (3, 2, False),
    (3, 3, True),
    (1, 1, True),
    (5, 10, True),
])
def test_reasoning_interval_scenarios(interval, steps, should_reason):
    """Test various reasoning interval scenarios."""
    agent = MagicMock()
    agent.reasoning = True
    agent.reasoning_interval = interval
    agent.adaptive_reasoning = False
    
    executor = CrewAgentExecutor(
        llm=MagicMock(),
        task=MagicMock(),
        crew=MagicMock(),
        agent=agent,
        prompt={},
        max_iter=10,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock()
    )
    
    executor.steps_since_reasoning = steps
    assert executor._should_trigger_reasoning() is should_reason

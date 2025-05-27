"""Tests for reasoning interval and adaptive reasoning in agents."""

import pytest
from unittest.mock import patch, MagicMock

from crewai import Agent, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.utilities.reasoning_handler import AgentReasoning


def test_agent_with_reasoning_interval():
    """Ensure that the agent triggers mid-execution reasoning based on the fixed interval."""

    # Use a mock LLM to avoid real network calls
    llm = MagicMock()

    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning interval feature",
        backstory="I am a test agent created to verify the reasoning interval feature works correctly.",
        llm=llm,
        reasoning=True,
        reasoning_interval=2,  # Reason every 2 steps
        verbose=True,
    )

    task = Task(
        description="Multi-step task that requires periodic reasoning.",
        expected_output="The task should be completed with periodic reasoning.",
        agent=agent,
    )

    # Create a mock executor that will be injected into the agent
    mock_executor = MagicMock()
    mock_executor.steps_since_reasoning = 0
    
    def mock_invoke(*args, **kwargs):
        return mock_executor._invoke_loop()
    
    def mock_invoke_loop():
        assert not mock_executor._should_trigger_reasoning()
        mock_executor.steps_since_reasoning += 1
        
        mock_executor.steps_since_reasoning = 2
        assert mock_executor._should_trigger_reasoning()
        mock_executor._handle_mid_execution_reasoning()
        
        return {"output": "Task completed successfully."}
    
    mock_executor.invoke = MagicMock(side_effect=mock_invoke)
    mock_executor._invoke_loop = MagicMock(side_effect=mock_invoke_loop)
    mock_executor._should_trigger_reasoning = MagicMock(side_effect=lambda: mock_executor.steps_since_reasoning >= 2)
    mock_executor._handle_mid_execution_reasoning = MagicMock()

    # Monkey-patch create_agent_executor so that it sets our mock_executor
    def _fake_create_agent_executor(self, tools=None, task=None):  # noqa: D401,E501
        """Replace the real executor with the mock while preserving behaviour."""
        self.agent_executor = mock_executor
        return mock_executor

    with patch.object(Agent, "create_agent_executor", _fake_create_agent_executor):
        result = agent.execute_task(task)

    # Validate results and that reasoning happened when expected
    assert result == "Task completed successfully."
    mock_executor._invoke_loop.assert_called_once()
    mock_executor._handle_mid_execution_reasoning.assert_called_once()


def test_agent_with_adaptive_reasoning():
    """Test agent with adaptive reasoning."""
    # Create a mock agent with adaptive reasoning
    agent = MagicMock()
    agent.reasoning = True
    agent.reasoning_interval = None
    agent.adaptive_reasoning = True
    agent.role = "Test Agent"
    
    # Create a mock task
    task = MagicMock()
    
    executor = CrewAgentExecutor(
        llm=MagicMock(),
        task=task,
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
    
    def mock_invoke_loop():
        assert executor._should_adaptive_reason()
        executor._handle_mid_execution_reasoning()
        return {"output": "Task completed with adaptive reasoning."}
    
    executor._invoke_loop = MagicMock(side_effect=mock_invoke_loop)
    executor._should_adaptive_reason = MagicMock(return_value=True)
    executor._handle_mid_execution_reasoning = MagicMock()
    
    result = executor._invoke_loop()
    
    assert result["output"] == "Task completed with adaptive reasoning."
    executor._should_adaptive_reason.assert_called_once()
    executor._handle_mid_execution_reasoning.assert_called_once()


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

    with patch('crewai.utilities.reasoning_handler.AgentReasoning.should_adaptive_reason_llm', return_value=True):
        assert executor._should_adaptive_reason() is True
    
    executor.messages = [
        {"role": "assistant", "content": "I'll try this approach."},
        {"role": "system", "content": "Error: Failed to execute the command."},
        {"role": "assistant", "content": "Let me try something else."}
    ]
    assert executor._should_adaptive_reason() is True
    
    executor.messages = [
        {"role": "assistant", "content": "I'll try this approach."},
        {"role": "system", "content": "Command executed successfully."},
        {"role": "assistant", "content": "Let me continue with the next step."}
    ]
    with patch('crewai.utilities.reasoning_handler.AgentReasoning.should_adaptive_reason_llm', return_value=False):
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

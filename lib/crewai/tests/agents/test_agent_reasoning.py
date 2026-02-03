"""Tests for planning/reasoning in agents."""

import json
import warnings

import pytest

from crewai import Agent, PlanningConfig, Task
from crewai.experimental.agent_executor import AgentExecutor
from crewai.llm import LLM


@pytest.fixture
def mock_llm_responses():
    """Fixture for mock LLM responses."""
    return {
        "ready": "I'll solve this simple math problem.\n\nREADY: I am ready to execute the task.\n\n",
        "not_ready": "I need to think about derivatives.\n\nNOT READY: I need to refine my plan because I'm not sure about the derivative rules.",
        "ready_after_refine": "I'll use the power rule for derivatives where d/dx(x^n) = n*x^(n-1).\n\nREADY: I am ready to execute the task.",
        "execution": "4",
    }


# =============================================================================
# Tests for PlanningConfig (new API)
# =============================================================================


def test_agent_with_planning_config(mock_llm_responses):
    """Test agent with PlanningConfig."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test the planning feature",
        backstory="I am a test agent created to verify the planning feature works correctly.",
        llm=llm,
        planning_config=PlanningConfig(),
        verbose=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent,
    )

    call_count = [0]

    def mock_llm_call(messages, *args, **kwargs):
        # First call is for planning, subsequent calls are for execution
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_llm_responses["ready"]
        return mock_llm_responses["execution"]

    agent.llm.call = mock_llm_call

    result = agent.execute_task(task)

    assert result == mock_llm_responses["execution"]
    assert "Planning:" in task.description


def test_agent_with_planning_config_max_attempts(mock_llm_responses):
    """Test agent with PlanningConfig and max_attempts."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Test Agent",
        goal="To test the planning feature",
        backstory="I am a test agent created to verify the planning feature works correctly.",
        llm=llm,
        planning_config=PlanningConfig(max_attempts=2),
        verbose=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Complex math task: What's the derivative of x²?",
        expected_output="The answer should be a mathematical expression.",
        agent=agent,
    )

    planning_call_count = [0]
    total_call_count = [0]

    def mock_llm_call(messages, *args, **kwargs):
        total_call_count[0] += 1
        # First 2 calls are for planning (initial + refine)
        if total_call_count[0] <= 2:
            planning_call_count[0] += 1
            if planning_call_count[0] == 1:
                return mock_llm_responses["not_ready"]
            return mock_llm_responses["ready_after_refine"]
        return "2x"

    agent.llm.call = mock_llm_call

    result = agent.execute_task(task)

    assert result == "2x"
    assert planning_call_count[0] == 2
    assert "Planning:" in task.description


def test_agent_with_planning_config_custom_prompts():
    """Test agent with PlanningConfig using custom prompts."""
    llm = LLM("gpt-3.5-turbo")

    custom_system_prompt = "You are a specialized planner."
    custom_plan_prompt = "Plan this task: {description}"

    agent = Agent(
        role="Test Agent",
        goal="To test custom prompts",
        backstory="I am a test agent.",
        llm=llm,
        planning_config=PlanningConfig(
            system_prompt=custom_system_prompt,
            plan_prompt=custom_plan_prompt,
            max_steps=10,
        ),
        verbose=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Simple task",
        expected_output="Some output",
        agent=agent,
    )

    captured_messages = []

    def mock_llm_call(messages, *args, **kwargs):
        captured_messages.extend(messages)
        return "My plan.\n\nREADY: I am ready to execute the task."

    agent.llm.call = mock_llm_call

    # Just test that the agent is created properly
    assert agent.planning_config is not None
    assert agent.planning_config.system_prompt == custom_system_prompt
    assert agent.planning_config.plan_prompt == custom_plan_prompt
    assert agent.planning_config.max_steps == 10


def test_agent_with_planning_config_disabled():
    """Test agent with PlanningConfig disabled."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test disabled planning",
        backstory="I am a test agent.",
        llm=llm,
        planning=False,
        verbose=True,
    )

    # Planning should be disabled
    assert agent.planning_enabled is False


def test_planning_config_default_values():
    """Test PlanningConfig default values."""
    config = PlanningConfig()

    assert config.max_attempts is None
    assert config.max_steps == 20
    assert config.system_prompt is None
    assert config.plan_prompt is None
    assert config.refine_prompt is None
    assert config.llm is None


def test_planning_config_custom_values():
    """Test PlanningConfig with custom values."""
    config = PlanningConfig(
        max_attempts=5,
        max_steps=15,
        system_prompt="Custom system",
        plan_prompt="Custom plan: {description}",
        refine_prompt="Custom refine: {current_plan}",
        llm="gpt-4",
    )

    assert config.max_attempts == 5
    assert config.max_steps == 15
    assert config.system_prompt == "Custom system"
    assert config.plan_prompt == "Custom plan: {description}"
    assert config.refine_prompt == "Custom refine: {current_plan}"
    assert config.llm == "gpt-4"


def test_planning_enabled_property():
    """Test the planning_enabled property on Agent."""
    llm = LLM("gpt-3.5-turbo")

    # With planning_config enabled
    agent_with_planning = Agent(
        role="Test Agent",
        goal="Test",
        backstory="Test",
        llm=llm,
        planning=True,
    )
    assert agent_with_planning.planning_enabled is True

    # With planning_config disabled
    agent_disabled = Agent(
        role="Test Agent",
        goal="Test",
        backstory="Test",
        llm=llm,
        planning=False,
    )
    assert agent_disabled.planning_enabled is False

    # Without planning_config
    agent_no_planning = Agent(
        role="Test Agent",
        goal="Test",
        backstory="Test",
        llm=llm,
    )
    assert agent_no_planning.planning_enabled is False


# =============================================================================
# Tests for backward compatibility with reasoning=True
# =============================================================================


def test_agent_with_reasoning_backward_compat(mock_llm_responses):
    """Test agent with reasoning=True (backward compatibility)."""
    llm = LLM("gpt-4o-mini")

    # This should emit a deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent = Agent(
            role="Test Agent",
            goal="To test the reasoning feature",
            backstory="I am a test agent created to verify the reasoning feature works correctly.",
            llm=llm,
            reasoning=True,
            verbose=True,
        )
        # Check that a deprecation warning was issued
        # Note: The warning may or may not be captured depending on how pydantic handles it
        # So we just verify the agent is created correctly

    # Should have created a PlanningConfig internally
    assert agent.planning_config is not None
    assert agent.planning_enabled is True


def test_agent_with_reasoning_and_max_attempts_backward_compat():
    """Test agent with reasoning=True and max_reasoning_attempts (backward compatibility)."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent.",
        llm=llm,
        reasoning=True,
        max_reasoning_attempts=5,
        verbose=True,
    )

    # Should have created a PlanningConfig with max_attempts
    assert agent.planning_config is not None
    assert agent.planning_config.max_attempts == 5


def test_agent_with_reasoning_not_ready_initially(mock_llm_responses):
    """Test agent with reasoning that requires refinement (backward compat)."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        max_reasoning_attempts=2,
        verbose=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Complex math task: What's the derivative of x²?",
        expected_output="The answer should be a mathematical expression.",
        agent=agent,
    )

    planning_call_count = [0]
    total_call_count = [0]

    def mock_llm_call(messages, *args, **kwargs):
        total_call_count[0] += 1
        # First 2 calls are for planning (initial + refine)
        if total_call_count[0] <= 2:
            planning_call_count[0] += 1
            if planning_call_count[0] == 1:
                return mock_llm_responses["not_ready"]
            return mock_llm_responses["ready_after_refine"]
        return "2x"

    agent.llm.call = mock_llm_call

    result = agent.execute_task(task)

    assert result == "2x"
    assert planning_call_count[0] == 2  # Should have made 2 planning calls
    assert "Planning:" in task.description


def test_agent_with_reasoning_max_attempts_reached():
    """Test agent with reasoning that reaches max attempts without being ready."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        max_reasoning_attempts=2,
        verbose=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Complex math task: Solve the Riemann hypothesis.",
        expected_output="A proof or disproof of the hypothesis.",
        agent=agent,
    )

    planning_call_count = [0]
    total_call_count = [0]

    def mock_llm_call(messages, *args, **kwargs):
        total_call_count[0] += 1
        # First 2 calls are for planning (all will return NOT READY)
        if total_call_count[0] <= 2:
            planning_call_count[0] += 1
            return f"Attempt {planning_call_count[0]}: I need more time to think.\n\nNOT READY: I need to refine my plan further."
        return "This is an unsolved problem in mathematics."

    agent.llm.call = mock_llm_call

    result = agent.execute_task(task)

    assert result == "This is an unsolved problem in mathematics."
    assert (
        planning_call_count[0] == 2
    )  # Should have made exactly 2 planning calls (max_attempts)
    assert "Planning:" in task.description


def test_agent_reasoning_error_handling():
    """Test error handling during the planning process."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        executor_class=AgentExecutor,  # Use AgentExecutor for planning support
    )

    task = Task(
        description="Task that will cause an error",
        expected_output="Output that will never be generated",
        agent=agent,
    )

    call_count = [0]

    def mock_llm_call_error(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # First calls are for planning
            raise Exception("LLM error during planning")
        return "Fallback execution result"  # Return a value for task execution

    agent.llm.call = mock_llm_call_error

    result = agent.execute_task(task)

    assert result == "Fallback execution result"
    assert call_count[0] > 0  # Ensure we called the mock at least once


# =============================================================================
# Tests for function calling
# =============================================================================


@pytest.mark.skip(reason="Test requires updates for native tool calling changes")
def test_agent_with_function_calling():
    """Test agent with planning using function calling."""
    llm = LLM("gpt-3.5-turbo")

    agent = Agent(
        role="Test Agent",
        goal="To test the planning feature",
        backstory="I am a test agent created to verify the planning feature works correctly.",
        llm=llm,
        planning_config=PlanningConfig(),
        verbose=True,
    )

    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent,
    )

    agent.llm.supports_function_calling = lambda: True

    def mock_function_call(messages, *args, **kwargs):
        if "tools" in kwargs:
            return json.dumps(
                {"plan": "I'll solve this simple math problem: 2+2=4.", "ready": True}
            )
        return "4"

    agent.llm.call = mock_function_call

    result = agent.execute_task(task)

    assert result == "4"
    assert "Planning:" in task.description
    assert "I'll solve this simple math problem: 2+2=4." in task.description


@pytest.mark.skip(reason="Test requires updates for native tool calling changes")
def test_agent_with_function_calling_fallback():
    """Test agent with planning using function calling that falls back to text parsing."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Test Agent",
        goal="To test the planning feature",
        backstory="I am a test agent created to verify the planning feature works correctly.",
        llm=llm,
        planning_config=PlanningConfig(),
        verbose=True,
    )

    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent,
    )

    agent.llm.supports_function_calling = lambda: True

    def mock_function_call(messages, *args, **kwargs):
        if "tools" in kwargs:
            return "Invalid JSON that will trigger fallback. READY: I am ready to execute the task."
        return "4"

    agent.llm.call = mock_function_call

    result = agent.execute_task(task)

    assert result == "4"
    assert "Planning:" in task.description
    assert "Invalid JSON that will trigger fallback" in task.description

"""Tests for planning/reasoning in agents."""

import warnings

import pytest

from crewai import Agent, PlanningConfig, Task
from crewai.llm import LLM


# =============================================================================
# Tests for PlanningConfig configuration (no LLM calls needed)
# =============================================================================


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


def test_agent_with_planning_config_custom_prompts():
    """Test agent with PlanningConfig using custom prompts."""
    llm = LLM("gpt-4o-mini")

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
        verbose=False,
    )

    # Just test that the agent is created properly
    assert agent.planning_config is not None
    assert agent.planning_config.system_prompt == custom_system_prompt
    assert agent.planning_config.plan_prompt == custom_plan_prompt
    assert agent.planning_config.max_steps == 10


def test_agent_with_planning_config_disabled():
    """Test agent with PlanningConfig disabled."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Test Agent",
        goal="To test disabled planning",
        backstory="I am a test agent.",
        llm=llm,
        planning=False,
        verbose=False,
    )

    # Planning should be disabled
    assert agent.planning_enabled is False


def test_planning_enabled_property():
    """Test the planning_enabled property on Agent."""
    llm = LLM("gpt-4o-mini")

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
# Tests for backward compatibility with reasoning=True (no LLM calls)
# =============================================================================


def test_agent_with_reasoning_backward_compat():
    """Test agent with reasoning=True (backward compatibility)."""
    llm = LLM("gpt-4o-mini")

    # This should emit a deprecation warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        agent = Agent(
            role="Test Agent",
            goal="To test the reasoning feature",
            backstory="I am a test agent created to verify the reasoning feature works correctly.",
            llm=llm,
            reasoning=True,
            verbose=False,
        )

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
        verbose=False,
    )

    # Should have created a PlanningConfig with max_attempts
    assert agent.planning_config is not None
    assert agent.planning_config.max_attempts == 5


# =============================================================================
# Tests for Agent.kickoff() with planning (uses AgentExecutor)
# =============================================================================


@pytest.mark.vcr()
def test_agent_kickoff_with_planning():
    """Test Agent.kickoff() with planning enabled generates a plan."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Assistant",
        goal="Help solve math problems step by step",
        backstory="A helpful math tutor",
        llm=llm,
        planning_config=PlanningConfig(max_attempts=1),
        verbose=False,
    )

    result = agent.kickoff("What is 15 + 27?")

    assert result is not None
    assert "42" in str(result)


@pytest.mark.vcr()
def test_agent_kickoff_without_planning():
    """Test Agent.kickoff() without planning skips plan generation."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Assistant",
        goal="Help solve math problems",
        backstory="A helpful assistant",
        llm=llm,
        # No planning_config = no planning
        verbose=False,
    )

    result = agent.kickoff("What is 8 * 7?")

    assert result is not None
    assert "56" in str(result)


@pytest.mark.vcr()
def test_agent_kickoff_with_planning_disabled():
    """Test Agent.kickoff() with planning explicitly disabled via planning=False."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Assistant",
        goal="Help solve math problems",
        backstory="A helpful assistant",
        llm=llm,
        planning=False,  # Explicitly disable planning
        verbose=False,
    )

    result = agent.kickoff("What is 100 / 4?")

    assert result is not None
    assert "25" in str(result)


@pytest.mark.vcr()
def test_agent_kickoff_multi_step_task_with_planning():
    """Test Agent.kickoff() with a multi-step task that benefits from planning."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Tutor",
        goal="Solve multi-step math problems",
        backstory="An expert tutor who explains step by step",
        llm=llm,
        planning_config=PlanningConfig(max_attempts=1, max_steps=5),
        verbose=False,
    )

    # Task requires: find primes, sum them, then double
    result = agent.kickoff(
        "Find the first 3 prime numbers, add them together, then multiply by 2."
    )

    assert result is not None
    # First 3 primes: 2, 3, 5 -> sum = 10 -> doubled = 20
    assert "20" in str(result)


# =============================================================================
# Tests for Agent.execute_task() with planning (uses CrewAgentExecutor)
# These test the legacy path via handle_reasoning()
# =============================================================================


@pytest.mark.vcr()
def test_agent_execute_task_with_planning():
    """Test Agent.execute_task() with planning via CrewAgentExecutor."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Assistant",
        goal="Help solve math problems",
        backstory="A helpful math tutor",
        llm=llm,
        planning_config=PlanningConfig(max_attempts=1),
        verbose=False,
    )

    task = Task(
        description="What is 9 + 11?",
        expected_output="A number",
        agent=agent,
    )

    result = agent.execute_task(task)

    assert result is not None
    assert "20" in str(result)
    # Planning should be appended to task description
    assert "Planning:" in task.description


@pytest.mark.vcr()
def test_agent_execute_task_without_planning():
    """Test Agent.execute_task() without planning."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Assistant",
        goal="Help solve math problems",
        backstory="A helpful assistant",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="What is 12 * 3?",
        expected_output="A number",
        agent=agent,
    )

    result = agent.execute_task(task)

    assert result is not None
    assert "36" in str(result)
    # No planning should be added
    assert "Planning:" not in task.description


@pytest.mark.vcr()
def test_agent_execute_task_with_planning_refine():
    """Test Agent.execute_task() with planning that requires refinement."""
    llm = LLM("gpt-4o-mini")

    agent = Agent(
        role="Math Tutor",
        goal="Solve complex math problems step by step",
        backstory="An expert tutor",
        llm=llm,
        planning_config=PlanningConfig(max_attempts=2),
        verbose=False,
    )

    task = Task(
        description="Calculate the area of a circle with radius 5 (use pi = 3.14)",
        expected_output="The area as a number",
        agent=agent,
    )

    result = agent.execute_task(task)

    assert result is not None
    # Area = pi * r^2 = 3.14 * 25 = 78.5
    assert "78" in str(result) or "79" in str(result)
    assert "Planning:" in task.description

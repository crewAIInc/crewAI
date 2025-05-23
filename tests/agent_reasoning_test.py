"""Tests for reasoning in agents."""

import json
import pytest

from crewai import Agent, Task
from crewai.llm import LLM
from crewai.utilities.reasoning_handler import AgentReasoning


@pytest.fixture
def mock_llm_responses():
    """Fixture for mock LLM responses."""
    return {
        "ready": "I'll solve this simple math problem.\n\nREADY: I am ready to execute the task.\n\n",
        "not_ready": "I need to think about derivatives.\n\nNOT READY: I need to refine my plan because I'm not sure about the derivative rules.",
        "ready_after_refine": "I'll use the power rule for derivatives where d/dx(x^n) = n*x^(n-1).\n\nREADY: I am ready to execute the task.",
        "execution": "4"
    }


def test_agent_with_reasoning(mock_llm_responses):
    """Test agent with reasoning."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        verbose=True
    )
    
    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent
    )
    
    agent.llm.call = lambda messages, *args, **kwargs: (
        mock_llm_responses["ready"]
        if any("create a detailed plan" in msg.get("content", "") for msg in messages)
        else mock_llm_responses["execution"]
    )
    
    result = agent.execute_task(task)
    
    assert result == mock_llm_responses["execution"]
    assert "Reasoning Plan:" in task.description


def test_agent_with_reasoning_not_ready_initially(mock_llm_responses):
    """Test agent with reasoning that requires refinement."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        max_reasoning_attempts=2,
        verbose=True
    )
    
    task = Task(
        description="Complex math task: What's the derivative of x²?",
        expected_output="The answer should be a mathematical expression.",
        agent=agent
    )
    
    call_count = [0]
    
    def mock_llm_call(messages, *args, **kwargs):
        if any("create a detailed plan" in msg.get("content", "") for msg in messages) or any("refine your plan" in msg.get("content", "") for msg in messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_llm_responses["not_ready"]
            else:
                return mock_llm_responses["ready_after_refine"]
        else:
            return "2x"
    
    agent.llm.call = mock_llm_call
    
    result = agent.execute_task(task)
    
    assert result == "2x"
    assert call_count[0] == 2  # Should have made 2 reasoning calls
    assert "Reasoning Plan:" in task.description


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
        verbose=True
    )
    
    task = Task(
        description="Complex math task: Solve the Riemann hypothesis.",
        expected_output="A proof or disproof of the hypothesis.",
        agent=agent
    )
    
    call_count = [0]
    
    def mock_llm_call(messages, *args, **kwargs):
        if any("create a detailed plan" in msg.get("content", "") for msg in messages) or any("refine your plan" in msg.get("content", "") for msg in messages):
            call_count[0] += 1
            return f"Attempt {call_count[0]}: I need more time to think.\n\nNOT READY: I need to refine my plan further."
        else:
            return "This is an unsolved problem in mathematics."
    
    agent.llm.call = mock_llm_call
    
    result = agent.execute_task(task)
    
    assert result == "This is an unsolved problem in mathematics."
    assert call_count[0] == 2  # Should have made exactly 2 reasoning calls (max_attempts)
    assert "Reasoning Plan:" in task.description


def test_agent_reasoning_input_validation():
    """Test input validation in AgentReasoning."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True
    )
    
    with pytest.raises(ValueError, match="Both task and agent must be provided"):
        AgentReasoning(task=None, agent=agent)
    
    task = Task(
        description="Simple task",
        expected_output="Simple output"
    )
    with pytest.raises(ValueError, match="Both task and agent must be provided"):
        AgentReasoning(task=task, agent=None)


def test_agent_reasoning_error_handling():
    """Test error handling during the reasoning process."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True
    )
    
    task = Task(
        description="Task that will cause an error",
        expected_output="Output that will never be generated",
        agent=agent
    )
    
    call_count = [0]
    
    def mock_llm_call_error(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # First calls are for reasoning
            raise Exception("LLM error during reasoning")
        return "Fallback execution result"  # Return a value for task execution
    
    agent.llm.call = mock_llm_call_error
    
    result = agent.execute_task(task)
    
    assert result == "Fallback execution result"
    assert call_count[0] > 2  # Ensure we called the mock multiple times


def test_agent_with_function_calling():
    """Test agent with reasoning using function calling."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        verbose=True
    )
    
    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent
    )
    
    agent.llm.supports_function_calling = lambda: True
    
    def mock_function_call(messages, *args, **kwargs):
        if "tools" in kwargs:
            return json.dumps({
                "plan": "I'll solve this simple math problem: 2+2=4.",
                "ready": True
            })
        else:
            return "4"
    
    agent.llm.call = mock_function_call
    
    result = agent.execute_task(task)
    
    assert result == "4"
    assert "Reasoning Plan:" in task.description
    assert "I'll solve this simple math problem: 2+2=4." in task.description


def test_agent_with_function_calling_fallback():
    """Test agent with reasoning using function calling that falls back to text parsing."""
    llm = LLM("gpt-3.5-turbo")
    
    agent = Agent(
        role="Test Agent",
        goal="To test the reasoning feature",
        backstory="I am a test agent created to verify the reasoning feature works correctly.",
        llm=llm,
        reasoning=True,
        verbose=True
    )
    
    task = Task(
        description="Simple math task: What's 2+2?",
        expected_output="The answer should be a number.",
        agent=agent
    )
    
    agent.llm.supports_function_calling = lambda: True
    
    def mock_function_call(messages, *args, **kwargs):
        if "tools" in kwargs:
            return "Invalid JSON that will trigger fallback. READY: I am ready to execute the task."
        else:
            return "4"
    
    agent.llm.call = mock_function_call
    
    result = agent.execute_task(task)
    
    assert result == "4"
    assert "Reasoning Plan:" in task.description
    assert "Invalid JSON that will trigger fallback" in task.description

import pytest
from unittest.mock import MagicMock

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.task import Task
from crewai.utilities.evaluators.crew_evaluator_handler import CrewEvaluator

@pytest.mark.vcr()
def test_crew_test_with_custom_llm():
    """Test Crew.test() with both string model name and LLM instance."""

    # Setup
    agent = Agent(
        role="test",
        goal="test",
        backstory="test",
        llm=LLM(model="gpt-4"),
    )
    task = Task(
        description="test",
        expected_output="test output",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])

    # Test with string model name
    crew.test(n_iterations=1, llm="gpt-4")

    # Test with LLM instance
    custom_llm = LLM(model="gpt-4")
    crew.test(n_iterations=1, llm=custom_llm)

    # Test backward compatibility
    crew.test(n_iterations=1, openai_model_name="gpt-4")

    # Test error when neither parameter is provided
    with pytest.raises(ValueError, match="Either llm or openai_model_name must be provided"):
        crew.test(n_iterations=1)

def test_crew_evaluator_with_custom_llm():
    # Setup
    agent = Agent(
        role="test",
        goal="test",
        backstory="test",
        llm=LLM(model="gpt-4"),
    )
    task = Task(
        description="test",
        expected_output="test output",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task])

    # Test with string model name
    evaluator = CrewEvaluator(crew, "gpt-4")
    assert isinstance(evaluator.llm, LLM)
    assert evaluator.llm.model == "gpt-4"

    # Test with LLM instance
    custom_llm = LLM(model="gpt-4")
    evaluator = CrewEvaluator(crew, custom_llm)
    assert evaluator.llm == custom_llm

    # Test that evaluator agent uses the correct LLM
    evaluator_agent = evaluator._evaluator_agent()
    assert evaluator_agent.llm == evaluator.llm

import pytest
from unittest.mock import MagicMock

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.task import Task
from crewai.utilities.evaluators.crew_evaluator_handler import CrewEvaluator

@pytest.mark.parametrize("model_input", [
    "gpt-4",  # Test string model name
    LLM(model="gpt-4"),  # Test LLM instance
])
def test_crew_test_with_custom_llm(model_input, mocker):
    # Mock LLM call to return valid JSON
    mocker.patch('crewai.llm.LLM.call', return_value='{"quality": 9.0}')

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

    # Test with provided model input
    crew.test(n_iterations=1, llm=model_input)

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

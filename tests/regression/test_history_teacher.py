import pytest
from crewai import Agent
from crewai_tools import SerperDevTool

from crewai.experimental.evaluation.testing import (
    assert_experiment_successfully,
    run_experiment,
)

@pytest.fixture
def history_teacher():
    search_tool = SerperDevTool()
    return Agent(
        role="History Educator",
        goal="Teach students about important historical events with clarity and context",
        backstory=(
            "As a renowned historian and educator, you have spent decades studying world history, "
            "from ancient civilizations to modern events. You are passionate about making history "
            "engaging and understandable for learners of all ages. Your mission is to educate, explain, "
            "and spark curiosity about the past."
        ),
        tools=[search_tool],
        verbose=True,
    )
def test_history_teacher(history_teacher):
    dataset = [
        {"inputs": {"messages": "How was the Battle of Waterloo?"}, "expected_score": 8}
    ]
    results = run_experiment(
        dataset=dataset, agents=[history_teacher], verbose=True
    )

    assert_experiment_successfully(results)

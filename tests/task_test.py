"""Test Agent creation and execution basic functionality."""

import json

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_core import ValidationError

from crewai import Agent, Crew, Process, Task
from crewai.tasks.task_output import TaskOutput


def test_task_tool_reflect_agent_tools():
    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 ideas.",
        agent=researcher,
    )

    assert task.tools == [fake_tool]


def test_task_tool_takes_precedence_over_agent_tools():
    from langchain.tools import tool

    @tool
    def fake_tool() -> None:
        "Fake tool"

    @tool
    def fake_task_tool() -> None:
        "Fake tool"

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[fake_tool],
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for an article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 ideas.",
        agent=researcher,
        tools=[fake_task_tool],
    )

    assert task.tools == [fake_task_tool]


def test_task_prompt_includes_expected_output():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        task.execute()
        execute.assert_called_once_with(task=task, context=None, tools=[])


def test_task_callback():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task_completed = MagicMock(return_value="done")

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
        callback=task_completed,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        task.execute()
        task_completed.assert_called_once_with(task.output)


def test_task_callback_returns_task_ouput():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task_completed = MagicMock(return_value="done")

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for an article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
        callback=task_completed,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "exported_ok"
        task.execute()
        # Ensure the callback is called with a TaskOutput object serialized to JSON
        task_completed.assert_called_once()
        callback_data = task_completed.call_args[0][0]

        # Check if callback_data is TaskOutput object or JSON string
        if isinstance(callback_data, TaskOutput):
            callback_data = json.dumps(callback_data.model_dump())

        assert isinstance(callback_data, str)
        output_dict = json.loads(callback_data)
        expected_output = {
            "description": task.description,
            "exported_output": "exported_ok",
            "raw_output": "exported_ok",
            "agent": researcher.role,
            "summary": "Give me a list of 5 interesting ideas to explore...",
        }
        assert output_dict == expected_output


def test_execute_with_agent():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
    )

    with patch.object(Agent, "execute_task", return_value="ok") as execute:
        task.execute(agent=researcher)
        execute.assert_called_once_with(task=task, context=None, tools=[])


def test_async_execution():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        async_execution=True,
        agent=researcher,
    )

    with patch.object(Agent, "execute_task", return_value="ok") as execute:
        task.execute(agent=researcher)
        execute.assert_called_once_with(task=task, context=None, tools=[])


def test_multiple_output_type_error():
    class Output(BaseModel):
        field: str

    with pytest.raises(ValidationError):
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            expected_output="Bullet point list of 5 interesting ideas.",
            output_json=Output,
            output_pydantic=Output,
        )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_pydantic():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_pydantic=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])
    result = crew.kickoff()
    assert isinstance(result, ScoreOutput)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_json=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])
    result = crew.kickoff()
    assert '{\n  "score": 4\n}' == result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_pydantic_to_another_task():
    from langchain_openai import ChatOpenAI

    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-4-0125-preview"),
        function_calling_llm=ChatOpenAI(model="gpt-3.5-turbo-0125"),
        verbose=True,
    )

    task1 = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_pydantic=ScoreOutput,
        agent=scorer,
    )

    task2 = Task(
        description="Given the score the title 'The impact of AI in the future of work' got, give me an integer score between 1-5 for the following title: 'Return of the Jedi', you MUST give it a score, use your best judgment",
        expected_output="The score of the title.",
        output_pydantic=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task1, task2], verbose=2)
    result = crew.kickoff()
    assert 5 == result.score


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json_to_another_task():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_json=ScoreOutput,
        agent=scorer,
    )

    task2 = Task(
        description="Given the score the title 'The impact of AI in the future of work' got, give me an integer score between 1-5 for the following title: 'Return of the Jedi'",
        expected_output="The score of the title.",
        output_json=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task1, task2])
    result = crew.kickoff()
    assert '{\n  "score": 3\n}' == result


@pytest.mark.vcr(filter_headers=["authorization"])
def test_save_task_output():
    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_file="score.json",
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])

    with patch.object(Task, "_save_file") as save_file:
        save_file.return_value = None
        crew.kickoff()
        save_file.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_save_task_json_output():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_file="score.json",
        output_json=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])

    with patch.object(Task, "_save_file") as save_file:
        save_file.return_value = None
        crew.kickoff()
        save_file.assert_called_once_with('{\n  "score": 4\n}')


@pytest.mark.vcr(filter_headers=["authorization"])
def test_save_task_pydantic_output():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        output_file="score.json",
        output_pydantic=ScoreOutput,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])

    with patch.object(Task, "_save_file") as save_file:
        save_file.return_value = None
        crew.kickoff()
        save_file.assert_called_once_with('{"score":4}')


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_delegations_for_hierarchical_process():
    from langchain_openai import ChatOpenAI

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
    )

    crew = Crew(
        agents=[scorer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4-0125-preview"),
    )

    with patch.object(Task, "increment_delegations") as increment_delegations:
        increment_delegations.return_value = None
        crew.kickoff()
        increment_delegations.assert_called_once


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_delegations_for_sequential_process():
    pass

    manager = Agent(
        role="Manager",
        goal="Coordinate scoring processes",
        backstory="You're great at delegating work about scoring.",
        allow_delegation=False,
    )

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work'",
        expected_output="The score of the title.",
        agent=manager,
    )

    crew = Crew(
        agents=[manager, scorer],
        tasks=[task],
        process=Process.sequential,
    )

    with patch.object(Task, "increment_delegations") as increment_delegations:
        increment_delegations.return_value = None
        crew.kickoff()
        increment_delegations.assert_called_once


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_tool_errors():
    from crewai_tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def scoring_examples() -> None:
        "Useful examples for scoring titles."
        raise Exception("Error")

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        tools=[scoring_examples],
    )

    task = Task(
        description="Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work', check examples to based your evaluation.",
        expected_output="The score of the title.",
    )

    crew = Crew(
        agents=[scorer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4-0125-preview"),
    )

    with patch.object(Task, "increment_tools_errors") as increment_tools_errors:
        increment_tools_errors.return_value = None
        crew.kickoff()
        increment_tools_errors.assert_called_once


def test_task_definition_based_on_dict():
    config = {
        "description": "Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work', check examples to based your evaluation.",
        "expected_output": "The score of the title.",
    }

    task = Task(config=config)

    assert task.description == config["description"]
    assert task.expected_output == config["expected_output"]
    assert task.agent is None


def test_interpolate_inputs():
    task = Task(
        description="Give me a list of 5 interesting ideas about {topic} to explore for an article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas about {topic}.",
    )

    task.interpolate_inputs(inputs={"topic": "AI"})
    assert (
        task.description
        == "Give me a list of 5 interesting ideas about AI to explore for an article, what makes them unique and interesting."
    )
    assert task.expected_output == "Bullet point list of 5 interesting ideas about AI."

    task.interpolate_inputs(inputs={"topic": "ML"})
    assert (
        task.description
        == "Give me a list of 5 interesting ideas about ML to explore for an article, what makes them unique and interesting."
    )
    assert task.expected_output == "Bullet point list of 5 interesting ideas about ML."

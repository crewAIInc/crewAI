"""Test Agent creation and execution basic functionality."""

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from crewai import Agent, Crew, Process, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.converter import Converter
from pydantic import BaseModel
from pydantic_core import ValidationError


def test_task_tool_reflect_agent_tools():
    from crewai_tools import tool

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
    from crewai_tools import tool

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
        task.execute_sync(agent=researcher)
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
        name="Brainstorm",
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 interesting ideas.",
        agent=researcher,
        callback=task_completed,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        task.execute_sync(agent=researcher)
        task_completed.assert_called_once_with(task.output)

        assert task.output.description == task.description
        assert task.output.expected_output == task.expected_output
        assert task.output.name == task.name


def test_task_callback_returns_task_output():
    from crewai.tasks.output_format import OutputFormat

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
        task.execute_sync(agent=researcher)
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
            "raw": "exported_ok",
            "pydantic": None,
            "json_dict": None,
            "agent": researcher.role,
            "summary": "Give me a list of 5 interesting ideas to explore...",
            "name": None,
            "expected_output": "Bullet point list of 5 interesting ideas.",
            "output_format": OutputFormat.RAW,
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
        task.execute_sync(agent=researcher)
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
        execution = task.execute_async(agent=researcher)
        result = execution.result()
        assert result.raw == "ok"
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
def test_output_pydantic_sequential():
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

    crew = Crew(agents=[scorer], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    assert isinstance(result.pydantic, ScoreOutput)
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_pydantic_hierarchical():
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

    crew = Crew(
        agents=[scorer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )
    result = crew.kickoff()
    assert isinstance(result.pydantic, ScoreOutput)
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json_sequential():
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
        output_file="score.json",
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    assert '{"score": 4}' == result.json
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json_hierarchical():
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

    crew = Crew(
        agents=[scorer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )
    result = crew.kickoff()
    assert result.json == '{"score": 4}'
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_json_property_without_output_json():
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
        output_pydantic=ScoreOutput,  # Using output_pydantic instead of output_json
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task], process=Process.sequential)
    result = crew.kickoff()

    with pytest.raises(ValueError) as excinfo:
        _ = result.json  # Attempt to access the json property

    assert "No JSON output found in the final task." in str(excinfo.value)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json_dict_sequential():
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

    crew = Crew(agents=[scorer], tasks=[task], process=Process.sequential)
    result = crew.kickoff()
    assert {"score": 4} == result.json_dict
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_json_dict_hierarchical():
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

    crew = Crew(
        agents=[scorer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )
    result = crew.kickoff()
    assert {"score": 4} == result.json_dict
    assert result.to_dict() == {"score": 4}


@pytest.mark.vcr(filter_headers=["authorization"])
def test_output_pydantic_to_another_task():
    class ScoreOutput(BaseModel):
        score: int

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=False,
        llm="gpt-4-0125-preview",
        function_calling_llm="gpt-3.5-turbo-0125",
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

    crew = Crew(agents=[scorer], tasks=[task1, task2], verbose=True)
    result = crew.kickoff()
    pydantic_result = result.pydantic
    assert isinstance(
        pydantic_result, ScoreOutput
    ), "Expected pydantic result to be of type ScoreOutput"
    assert pydantic_result.score == 5


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
    assert '{"score": 4}' == result.json


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
    crew.kickoff()

    output_file_exists = os.path.exists("score.json")
    assert output_file_exists
    assert {"score": 4} == json.loads(open("score.json").read())
    if output_file_exists:
        os.remove("score.json")


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
    crew.kickoff()

    output_file_exists = os.path.exists("score.json")
    assert output_file_exists
    assert {"score": 4} == json.loads(open("score.json").read())
    if output_file_exists:
        os.remove("score.json")


@pytest.mark.vcr(filter_headers=["authorization"])
def test_custom_converter_cls():
    class ScoreOutput(BaseModel):
        score: int

    class ScoreConverter(Converter):
        pass

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
        converter_cls=ScoreConverter,
        agent=scorer,
    )

    crew = Crew(agents=[scorer], tasks=[task])

    with patch.object(
        ScoreConverter, "to_pydantic", return_value=ScoreOutput(score=5)
    ) as mock_to_pydantic:
        crew.kickoff()
        mock_to_pydantic.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_delegations_for_hierarchical_process():
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
        manager_llm="gpt-4o",
    )

    with patch.object(Task, "increment_delegations") as increment_delegations:
        increment_delegations.return_value = None
        crew.kickoff()
        increment_delegations.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_delegations_for_sequential_process():
    manager = Agent(
        role="Manager",
        goal="Coordinate scoring processes",
        backstory="You're great at delegating work about scoring.",
        allow_delegation=True,
    )

    scorer = Agent(
        role="Scorer",
        goal="Score the title",
        backstory="You're an expert scorer, specialized in scoring titles.",
        allow_delegation=True,
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
        increment_delegations.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_increment_tool_errors():
    from crewai_tools import tool

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
        manager_llm="gpt-4-0125-preview",
    )

    with patch.object(Task, "increment_tools_errors") as increment_tools_errors:
        increment_tools_errors.return_value = None
        crew.kickoff()
        assert len(increment_tools_errors.mock_calls) == 12


def test_task_definition_based_on_dict():
    config = {
        "description": "Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work', check examples to based your evaluation.",
        "expected_output": "The score of the title.",
    }

    task = Task(**config)

    assert task.description == config["description"]
    assert task.expected_output == config["expected_output"]
    assert task.agent is None


def test_conditional_task_definition_based_on_dict():
    config = {
        "description": "Give me an integer score between 1-5 for the following title: 'The impact of AI in the future of work', check examples to based your evaluation.",
        "expected_output": "The score of the title.",
    }

    task = ConditionalTask(**config, condition=lambda x: True)

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


def test_task_output_str_with_pydantic():
    from crewai.tasks.output_format import OutputFormat

    class ScoreOutput(BaseModel):
        score: int

    score_output = ScoreOutput(score=4)
    task_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        pydantic=score_output,
        output_format=OutputFormat.PYDANTIC,
    )

    assert str(task_output) == str(score_output)


def test_task_output_str_with_json_dict():
    from crewai.tasks.output_format import OutputFormat

    json_dict = {"score": 4}
    task_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        json_dict=json_dict,
        output_format=OutputFormat.JSON,
    )

    assert str(task_output) == str(json_dict)


def test_task_output_str_with_raw():
    from crewai.tasks.output_format import OutputFormat

    raw_output = "Raw task output"
    task_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        raw=raw_output,
        output_format=OutputFormat.RAW,
    )

    assert str(task_output) == raw_output


def test_task_output_str_with_pydantic_and_json_dict():
    from crewai.tasks.output_format import OutputFormat

    class ScoreOutput(BaseModel):
        score: int

    score_output = ScoreOutput(score=4)
    json_dict = {"score": 4}
    task_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        pydantic=score_output,
        json_dict=json_dict,
        output_format=OutputFormat.PYDANTIC,
    )

    # When both pydantic and json_dict are present, pydantic should take precedence
    assert str(task_output) == str(score_output)


def test_task_output_str_with_none():
    from crewai.tasks.output_format import OutputFormat

    task_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        output_format=OutputFormat.RAW,
    )

    assert str(task_output) == ""


def test_key():
    original_description = "Give me a list of 5 interesting ideas about {topic} to explore for an article, what makes them unique and interesting."
    original_expected_output = "Bullet point list of 5 interesting ideas about {topic}."
    task = Task(
        description=original_description,
        expected_output=original_expected_output,
    )
    hash = hashlib.md5(
        f"{original_description}|{original_expected_output}".encode()
    ).hexdigest()

    assert task.key == hash, "The key should be the hash of the description."

    task.interpolate_inputs(inputs={"topic": "AI"})
    assert (
        task.key == hash
    ), "The key should be the hash of the non-interpolated description."

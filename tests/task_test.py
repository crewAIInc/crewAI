"""Test Agent creation and execution basic functionality."""

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic_core import ValidationError

from crewai import Agent, Crew, Process, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.converter import Converter


def test_task_tool_reflect_agent_tools():
    from crewai.tools import tool

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
    from crewai.tools import tool

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
    from crewai.tools import tool

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
        assert len(increment_tools_errors.mock_calls) > 0


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
        output_file="/tmp/{topic}/output_{date}.txt",
    )

    task.interpolate_inputs_and_add_conversation_history(
        inputs={"topic": "AI", "date": "2025"}
    )
    assert (
        task.description
        == "Give me a list of 5 interesting ideas about AI to explore for an article, what makes them unique and interesting."
    )
    assert task.expected_output == "Bullet point list of 5 interesting ideas about AI."
    assert task.output_file == "/tmp/AI/output_2025.txt"

    task.interpolate_inputs_and_add_conversation_history(
        inputs={"topic": "ML", "date": "2025"}
    )
    assert (
        task.description
        == "Give me a list of 5 interesting ideas about ML to explore for an article, what makes them unique and interesting."
    )
    assert task.expected_output == "Bullet point list of 5 interesting ideas about ML."
    assert task.output_file == "/tmp/ML/output_2025.txt"


def test_interpolate_only():
    """Test the interpolate_only method for various scenarios including JSON structure preservation."""
    task = Task(
        description="Unused in this test", expected_output="Unused in this test"
    )

    # Test JSON structure preservation
    json_string = '{"info": "Look at {placeholder}", "nested": {"val": "{nestedVal}"}}'
    result = task.interpolate_only(
        input_string=json_string,
        inputs={"placeholder": "the data", "nestedVal": "something else"},
    )
    assert '"info": "Look at the data"' in result
    assert '"val": "something else"' in result
    assert "{placeholder}" not in result
    assert "{nestedVal}" not in result

    # Test normal string interpolation
    normal_string = "Hello {name}, welcome to {place}!"
    result = task.interpolate_only(
        input_string=normal_string, inputs={"name": "John", "place": "CrewAI"}
    )
    assert result == "Hello John, welcome to CrewAI!"

    # Test empty string
    result = task.interpolate_only(input_string="", inputs={"unused": "value"})
    assert result == ""

    # Test string with no placeholders
    no_placeholders = "Hello, this is a test"
    result = task.interpolate_only(
        input_string=no_placeholders, inputs={"unused": "value"}
    )
    assert result == no_placeholders


def test_interpolate_only_with_dict_inside_expected_output():
    """Test the interpolate_only method for various scenarios including JSON structure preservation."""
    task = Task(
        description="Unused in this test",
        expected_output="Unused in this test: {questions}",
    )

    json_string = '{"questions": {"main_question": "What is the user\'s name?", "secondary_question": "What is the user\'s age?"}}'
    result = task.interpolate_only(
        input_string=json_string,
        inputs={
            "questions": {
                "main_question": "What is the user's name?",
                "secondary_question": "What is the user's age?",
            }
        },
    )
    assert '"main_question": "What is the user\'s name?"' in result
    assert '"secondary_question": "What is the user\'s age?"' in result
    assert result == json_string

    normal_string = "Hello {name}, welcome to {place}!"
    result = task.interpolate_only(
        input_string=normal_string, inputs={"name": "John", "place": "CrewAI"}
    )
    assert result == "Hello John, welcome to CrewAI!"

    result = task.interpolate_only(input_string="", inputs={"unused": "value"})
    assert result == ""

    no_placeholders = "Hello, this is a test"
    result = task.interpolate_only(
        input_string=no_placeholders, inputs={"unused": "value"}
    )
    assert result == no_placeholders


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

    task.interpolate_inputs_and_add_conversation_history(inputs={"topic": "AI"})
    assert (
        task.key == hash
    ), "The key should be the hash of the non-interpolated description."


def test_output_file_validation():
    """Test output file path validation."""
    # Valid paths
    assert (
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="output.txt",
        ).output_file
        == "output.txt"
    )
    assert (
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="/tmp/output.txt",
        ).output_file
        == "tmp/output.txt"
    )
    assert (
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="{dir}/output_{date}.txt",
        ).output_file
        == "{dir}/output_{date}.txt"
    )

    # Invalid paths
    with pytest.raises(ValueError, match="Path traversal"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="../output.txt",
        )
    with pytest.raises(ValueError, match="Path traversal"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="folder/../output.txt",
        )
    with pytest.raises(ValueError, match="Shell special characters"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="output.txt | rm -rf /",
        )
    with pytest.raises(ValueError, match="Shell expansion"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="~/output.txt",
        )
    with pytest.raises(ValueError, match="Shell expansion"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="$HOME/output.txt",
        )
    with pytest.raises(ValueError, match="Invalid template variable"):
        Task(
            description="Test task",
            expected_output="Test output",
            output_file="{invalid-name}/output.txt",
        )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_task_execution_times():
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

    assert task.start_time is None
    assert task.end_time is None
    assert task.execution_duration is None

    task.execute_sync(agent=researcher)

    assert task.start_time is not None
    assert task.end_time is not None
    assert task.execution_duration == (task.end_time - task.start_time).total_seconds()


def test_interpolate_with_list_of_strings():
    task = Task(
        description="Test list interpolation",
        expected_output="List: {items}",
    )

    # Test simple list of strings
    input_str = "Available items: {items}"
    inputs = {"items": ["apple", "banana", "cherry"]}
    result = task.interpolate_only(input_str, inputs)
    assert result == f"Available items: {inputs['items']}"

    # Test empty list
    empty_list_input = {"items": []}
    result = task.interpolate_only(input_str, empty_list_input)
    assert result == "Available items: []"


def test_interpolate_with_list_of_dicts():
    task = Task(
        description="Test list of dicts interpolation",
        expected_output="People: {people}",
    )

    input_data = {
        "people": [
            {"name": "Alice", "age": 30, "skills": ["Python", "AI"]},
            {"name": "Bob", "age": 25, "skills": ["Java", "Cloud"]},
        ]
    }
    result = task.interpolate_only("{people}", input_data)

    parsed_result = eval(result)
    assert isinstance(parsed_result, list)
    assert len(parsed_result) == 2
    assert parsed_result[0]["name"] == "Alice"
    assert parsed_result[0]["age"] == 30
    assert parsed_result[0]["skills"] == ["Python", "AI"]
    assert parsed_result[1]["name"] == "Bob"
    assert parsed_result[1]["age"] == 25
    assert parsed_result[1]["skills"] == ["Java", "Cloud"]


def test_interpolate_with_nested_structures():
    task = Task(
        description="Test nested structures",
        expected_output="Company: {company}",
    )

    input_data = {
        "company": {
            "name": "TechCorp",
            "departments": [
                {
                    "name": "Engineering",
                    "employees": 50,
                    "tools": ["Git", "Docker", "Kubernetes"],
                },
                {"name": "Sales", "employees": 20, "regions": {"north": 5, "south": 3}},
            ],
        }
    }
    result = task.interpolate_only("{company}", input_data)
    parsed = eval(result)

    assert parsed["name"] == "TechCorp"
    assert len(parsed["departments"]) == 2
    assert parsed["departments"][0]["tools"] == ["Git", "Docker", "Kubernetes"]
    assert parsed["departments"][1]["regions"]["north"] == 5


def test_interpolate_with_special_characters():
    task = Task(
        description="Test special characters in dicts",
        expected_output="Data: {special_data}",
    )

    input_data = {
        "special_data": {
            "quotes": """This has "double" and 'single' quotes""",
            "unicode": "文字化けテスト",
            "symbols": "!@#$%^&*()",
            "empty": "",
        }
    }
    result = task.interpolate_only("{special_data}", input_data)
    parsed = eval(result)

    assert parsed["quotes"] == """This has "double" and 'single' quotes"""
    assert parsed["unicode"] == "文字化けテスト"
    assert parsed["symbols"] == "!@#$%^&*()"
    assert parsed["empty"] == ""


def test_interpolate_mixed_types():
    task = Task(
        description="Test mixed type interpolation",
        expected_output="Mixed: {data}",
    )

    input_data = {
        "data": {
            "name": "Test Dataset",
            "samples": 1000,
            "features": ["age", "income", "location"],
            "metadata": {
                "source": "public",
                "validated": True,
                "tags": ["demo", "test", "temp"],
            },
        }
    }
    result = task.interpolate_only("{data}", input_data)
    parsed = eval(result)

    assert parsed["name"] == "Test Dataset"
    assert parsed["samples"] == 1000
    assert parsed["metadata"]["tags"] == ["demo", "test", "temp"]


def test_interpolate_complex_combination():
    task = Task(
        description="Test complex combination",
        expected_output="Report: {report}",
    )

    input_data = {
        "report": [
            {
                "month": "January",
                "metrics": {"sales": 15000, "expenses": 8000, "profit": 7000},
                "top_products": ["Product A", "Product B"],
            },
            {
                "month": "February",
                "metrics": {"sales": 18000, "expenses": 8500, "profit": 9500},
                "top_products": ["Product C", "Product D"],
            },
        ]
    }
    result = task.interpolate_only("{report}", input_data)
    parsed = eval(result)

    assert len(parsed) == 2
    assert parsed[0]["month"] == "January"
    assert parsed[1]["metrics"]["profit"] == 9500
    assert "Product D" in parsed[1]["top_products"]


def test_interpolate_invalid_type_validation():
    task = Task(
        description="Test invalid type validation",
        expected_output="Should never reach here",
    )

    # Test with invalid top-level type
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only("{data}", {"data": set()})  # type: ignore we are purposely testing this failure

    assert "Unsupported type set" in str(excinfo.value)

    # Test with invalid nested type
    invalid_nested = {
        "profile": {
            "name": "John",
            "age": 30,
            "tags": {"a", "b", "c"},  # Set is invalid
        }
    }
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only("{data}", {"data": invalid_nested})
    assert "Unsupported type set" in str(excinfo.value)


def test_interpolate_custom_object_validation():
    task = Task(
        description="Test custom object rejection",
        expected_output="Should never reach here",
    )

    class CustomObject:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return str(self.value)

    # Test with custom object at top level
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only("{obj}", {"obj": CustomObject(5)})  # type: ignore we are purposely testing this failure
    assert "Unsupported type CustomObject" in str(excinfo.value)

    # Test with nested custom object in dictionary
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only(
            "{data}", {"data": {"valid": 1, "invalid": CustomObject(5)}}
        )
    assert "Unsupported type CustomObject" in str(excinfo.value)

    # Test with nested custom object in list
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only("{data}", {"data": [1, "valid", CustomObject(5)]})
    assert "Unsupported type CustomObject" in str(excinfo.value)

    # Test with deeply nested custom object
    with pytest.raises(ValueError) as excinfo:
        task.interpolate_only(
            "{data}", {"data": {"level1": {"level2": [{"level3": CustomObject(5)}]}}}
        )
    assert "Unsupported type CustomObject" in str(excinfo.value)


def test_interpolate_valid_complex_types():
    task = Task(
        description="Test valid complex types",
        expected_output="Validation should pass",
    )

    # Valid complex structure
    valid_data = {
        "name": "Valid Dataset",
        "stats": {
            "count": 1000,
            "distribution": [0.2, 0.3, 0.5],
            "features": ["age", "income"],
            "nested": {"deep": [1, 2, 3], "deeper": {"a": 1, "b": 2.5}},
        },
    }

    # Should not raise any errors
    result = task.interpolate_only("{data}", {"data": valid_data})
    parsed = eval(result)
    assert parsed["name"] == "Valid Dataset"
    assert parsed["stats"]["nested"]["deeper"]["b"] == 2.5


def test_interpolate_edge_cases():
    task = Task(
        description="Test edge cases",
        expected_output="Edge case handling",
    )

    # Test empty dict and list
    assert task.interpolate_only("{}", {"data": {}}) == "{}"
    assert task.interpolate_only("[]", {"data": []}) == "[]"

    # Test numeric types
    assert task.interpolate_only("{num}", {"num": 42}) == "42"
    assert task.interpolate_only("{num}", {"num": 3.14}) == "3.14"

    # Test boolean values (valid JSON types)
    assert task.interpolate_only("{flag}", {"flag": True}) == "True"
    assert task.interpolate_only("{flag}", {"flag": False}) == "False"


def test_interpolate_valid_types():
    task = Task(
        description="Test valid types including null and boolean",
        expected_output="Should pass validation",
    )

    # Test with boolean and null values (valid JSON types)
    valid_data = {
        "name": "Test",
        "active": True,
        "deleted": False,
        "optional": None,
        "nested": {"flag": True, "empty": None},
    }

    result = task.interpolate_only("{data}", {"data": valid_data})
    parsed = eval(result)

    assert parsed["active"] is True
    assert parsed["deleted"] is False
    assert parsed["optional"] is None
    assert parsed["nested"]["flag"] is True
    assert parsed["nested"]["empty"] is None

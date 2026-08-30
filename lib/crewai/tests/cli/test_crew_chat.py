"""Tests for ``crewai.utilities.crew_chat`` startup-safety helpers."""

from unittest import mock

from crewai.utilities.crew_chat import (
    DEFAULT_CREW_DESCRIPTION,
    DEFAULT_INPUT_DESCRIPTION,
    generate_crew_chat_inputs,
    generate_crew_description_with_ai,
    generate_input_description_with_ai,
)


def _make_crew(
    *,
    task_description: str = "",
    expected_output: str = "",
    agent_role: str = "",
    agent_goal: str = "",
    agent_backstory: str = "",
    inputs: set[str] | None = None,
) -> mock.Mock:
    task = mock.Mock()
    task.description = task_description
    task.expected_output = expected_output

    agent = mock.Mock()
    agent.role = agent_role
    agent.goal = agent_goal
    agent.backstory = agent_backstory

    crew = mock.Mock()
    crew.tasks = [task]
    crew.agents = [agent]
    crew.fetch_inputs = mock.Mock(return_value=inputs or set())
    return crew


def test_generate_input_description_falls_back_on_llm_failure() -> None:
    crew = _make_crew(task_description="Summarize {topic} for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    description = generate_input_description_with_ai("topic", crew, chat_llm)

    assert description == DEFAULT_INPUT_DESCRIPTION
    chat_llm.call.assert_called_once()


def test_generate_crew_description_falls_back_on_llm_failure() -> None:
    crew = _make_crew(task_description="Summarize topic for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    description = generate_crew_description_with_ai(crew, chat_llm)

    assert description == DEFAULT_CREW_DESCRIPTION
    chat_llm.call.assert_called_once()


def test_generate_input_description_returns_llm_response_on_success() -> None:
    crew = _make_crew(task_description="Summarize {topic} for the team.")
    chat_llm = mock.Mock()
    chat_llm.call.return_value = "  the subject to summarize  "

    description = generate_input_description_with_ai("topic", crew, chat_llm)

    assert description == "the subject to summarize"


def test_generate_crew_chat_inputs_skips_llm_when_descriptions_disabled() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()

    chat_inputs = generate_crew_chat_inputs(
        crew, "demo-crew", chat_llm, generate_descriptions=False
    )

    assert chat_inputs.crew_name == "demo-crew"
    assert chat_inputs.crew_description == DEFAULT_CREW_DESCRIPTION
    assert len(chat_inputs.inputs) == 1
    assert chat_inputs.inputs[0].name == "topic"
    assert chat_inputs.inputs[0].description == DEFAULT_INPUT_DESCRIPTION
    chat_llm.call.assert_not_called()


def test_generate_crew_chat_inputs_uses_llm_by_default() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = ["the subject to summarize", "summarize topics"]

    chat_inputs = generate_crew_chat_inputs(crew, "demo-crew", chat_llm)

    assert chat_inputs.crew_description == "summarize topics"
    assert chat_inputs.inputs[0].description == "the subject to summarize"
    assert chat_llm.call.call_count == 2


def test_generate_crew_chat_inputs_falls_back_when_llm_fails_mid_run() -> None:
    crew = _make_crew(
        task_description="Summarize {topic} for the team.",
        inputs={"topic"},
    )
    chat_llm = mock.Mock()
    chat_llm.call.side_effect = RuntimeError("APIConnectionError")

    chat_inputs = generate_crew_chat_inputs(crew, "demo-crew", chat_llm)

    assert chat_inputs.crew_description == DEFAULT_CREW_DESCRIPTION
    assert chat_inputs.inputs[0].description == DEFAULT_INPUT_DESCRIPTION
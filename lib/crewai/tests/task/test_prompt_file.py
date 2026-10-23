"""Regression coverage for Crew.prompt_file in task prompts."""

import json
from pathlib import Path
from unittest.mock import patch

import crewai
from crewai import LLM, Agent, Crew, Process, Task
from crewai.utilities.i18n import I18N_DEFAULT
import pytest


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    prompts_path = Path(crewai.__file__).parent / "translations" / "en.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts["slices"]["expected_output"] = "CUSTOM OUTPUT: {expected_output}"
    prompts["slices"]["conversation_history_instruction"] = "CUSTOM HISTORY"
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(prompts), encoding="utf-8")
    return path


def make_crew(prompt_file: Path | None, process: Process) -> Crew:
    llm = LLM(model="openai/gpt-4o-mini", api_key="test-key")
    agent = Agent(
        role="Researcher", goal="Answer questions", backstory="A researcher", llm=llm
    )
    task = Task(
        description="Summarize a document",
        expected_output="A short summary",
        agent=agent if process == Process.sequential else None,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        process=process,
        manager_llm=llm if process == Process.hierarchical else None,
        prompt_file=str(prompt_file) if prompt_file else None,
        tracing=False,
    )


@pytest.mark.parametrize("process", [Process.sequential, Process.hierarchical])
def test_kickoff_uses_custom_task_prompt_without_inputs(
    prompt_file: Path, process: Process
) -> None:
    crew = make_crew(prompt_file, process)
    with patch.object(
        Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
    ):
        output = crew.kickoff()

    assert output.raw == "Summarize a document\nCUSTOM OUTPUT: A short summary"


@pytest.mark.parametrize("process", [Process.sequential, Process.hierarchical])
def test_kickoff_uses_custom_conversation_instruction(
    prompt_file: Path, process: Process
) -> None:
    crew = make_crew(prompt_file, process)
    inputs = {
        "crew_chat_messages": json.dumps(
            [{"role": "user", "content": "Discuss the findings"}]
        )
    }
    with patch.object(
        Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
    ):
        output = crew.kickoff(inputs=inputs)

    assert "CUSTOM HISTORY\n\nUser: Discuss the findings" in output.raw
    assert "CUSTOM OUTPUT: A short summary" in output.raw


@pytest.mark.asyncio
@pytest.mark.parametrize("process", [Process.sequential, Process.hierarchical])
async def test_native_async_kickoff_uses_custom_task_prompt(
    prompt_file: Path, process: Process
) -> None:
    crew = make_crew(prompt_file, process)
    with patch.object(
        Agent, "aexecute_task", side_effect=lambda task, **_: task.prompt()
    ):
        output = await crew.akickoff()

    assert output.raw == "Summarize a document\nCUSTOM OUTPUT: A short summary"


def test_kickoff_resolves_updated_prompt_file_and_restores_default(
    prompt_file: Path, tmp_path: Path
) -> None:
    crew = make_crew(prompt_file, Process.sequential)
    second_file = tmp_path / "second-prompts.json"
    second_file.write_text(
        prompt_file.read_text(encoding="utf-8").replace(
            "CUSTOM OUTPUT", "SECOND OUTPUT"
        ),
        encoding="utf-8",
    )
    with patch.object(
        Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
    ):
        first = crew.kickoff().raw
        crew.prompt_file = str(second_file)
        second = crew.kickoff().raw
        crew.prompt_file = None
        default = crew.kickoff().raw

    assert "CUSTOM OUTPUT: A short summary" in first
    assert "SECOND OUTPUT: A short summary" in second
    assert "CUSTOM OUTPUT" not in second
    assert default == "Summarize a document\n" + I18N_DEFAULT.slice(
        "expected_output"
    ).format(expected_output="A short summary")


def test_custom_task_prompt_does_not_leak_between_crews(prompt_file: Path) -> None:
    custom_crew = make_crew(prompt_file, Process.sequential)
    default_crew = make_crew(None, Process.sequential)
    with patch.object(
        Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
    ):
        custom = custom_crew.kickoff().raw
        default = default_crew.kickoff().raw
        repeated = custom_crew.kickoff().raw

    assert "CUSTOM OUTPUT: A short summary" in custom
    assert "CUSTOM OUTPUT" not in default
    assert repeated == custom


def test_standalone_task_keeps_default_prompt() -> None:
    task = Task(description="Summarize a document", expected_output="A short summary")
    assert task.prompt() == "Summarize a document\n" + I18N_DEFAULT.slice(
        "expected_output"
    ).format(expected_output="A short summary")


@pytest.mark.parametrize("process", [Process.sequential, Process.hierarchical])
@pytest.mark.parametrize("with_history", [False, True])
def test_replay_uses_custom_task_prompt_on_recreated_crew(
    prompt_file: Path, process: Process, with_history: bool
) -> None:
    """Replay resolves custom instructions even without an earlier kickoff."""
    previous_task = make_crew(prompt_file, process).tasks[0]
    inputs = (
        {
            "crew_chat_messages": json.dumps(
                [{"role": "user", "content": "Discuss the findings"}]
            )
        }
        if with_history
        else {}
    )
    stored_output = {
        "task_id": str(previous_task.id),
        "task_key": previous_task.key,
        "expected_output": previous_task.expected_output,
        "output": {"description": previous_task.description},
        "inputs": inputs,
    }
    crew = make_crew(prompt_file, process)
    with (
        patch(
            "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
            return_value=[stored_output],
        ),
        patch.object(
            Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
        ),
    ):
        output = crew.replay(str(previous_task.id))

    assert "CUSTOM OUTPUT: A short summary" in output.raw
    if with_history:
        assert "CUSTOM HISTORY\n\nUser: Discuss the findings" in output.raw


@pytest.mark.parametrize("restore_default", [False, True])
def test_replay_resolves_updated_prompt_file_after_kickoff(
    prompt_file: Path, tmp_path: Path, restore_default: bool
) -> None:
    """Replay uses the current prompt configuration rather than the last kickoff's."""
    crew = make_crew(prompt_file, Process.sequential)
    task = crew.tasks[0]
    stored_output = {
        "task_id": str(task.id),
        "task_key": task.key,
        "expected_output": task.expected_output,
        "output": {"description": task.description},
        "inputs": {},
    }
    second_file = tmp_path / "second-prompts.json"
    second_file.write_text(
        prompt_file.read_text(encoding="utf-8").replace(
            "CUSTOM OUTPUT", "SECOND OUTPUT"
        ),
        encoding="utf-8",
    )
    with (
        patch(
            "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
            return_value=[stored_output],
        ),
        patch.object(
            Agent, "execute_task", side_effect=lambda task, **_: task.prompt()
        ),
    ):
        assert "CUSTOM OUTPUT: A short summary" in crew.kickoff().raw
        crew.prompt_file = None if restore_default else str(second_file)
        output = crew.replay(str(task.id))

    expected = (
        I18N_DEFAULT.slice("expected_output").format(expected_output="A short summary")
        if restore_default
        else "SECOND OUTPUT: A short summary"
    )
    assert output.raw == "Summarize a document\n" + expected

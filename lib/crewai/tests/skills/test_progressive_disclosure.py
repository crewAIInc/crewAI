"""Regression tests for runtime skill progressive disclosure.

Run this focused test file with:

    uv run pytest lib/crewai/tests/skills/test_progressive_disclosure.py -q
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewai import Agent, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.llms.base_llm import BaseLLM
from crewai.skills.models import METADATA
from crewai.skills.parser import load_skill_metadata
from crewai.skills.tool import LoadSkillTool, create_skill_loader_tool
from crewai.tools.base_tool import BaseTool
from crewai.utilities.prompts import Prompts


class _ConflictingToolSchema(BaseModel):
    query: str = Field(default="")


class _ConflictingTool(BaseTool):
    """A user tool that already claims the skill loader's default name."""

    name: str = "load_skill"
    description: str = "Load something unrelated to skills."
    args_schema: type[BaseModel] = _ConflictingToolSchema

    def _run(self, query: str = "", **kwargs: Any) -> str:
        return "user tool result"


def _create_skill(
    parent: Path,
    name: str,
    description: str,
    instructions: str,
) -> None:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{instructions}",
        encoding="utf-8",
    )


class _SkillChoosingLLM(BaseLLM):
    """Small deterministic LLM that exercises the real agent tool loop."""

    def call(self, messages: Any, **kwargs: Any) -> str:
        rendered = str(messages)
        if "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS" in rendered:
            return "Thought: I loaded the review skill.\nFinal Answer: python loaded"
        if "TRAVEL_PRIVATE_INSTRUCTIONS" in rendered:
            return "Thought: I loaded the travel skill.\nFinal Answer: travel loaded"
        if "Review this Python change" in rendered:
            return (
                "Thought: The Python review skill applies.\n"
                "Action: load_skill\n"
                'Action Input: {"skill_name": "python-review"}'
            )
        if "Plan a weekend trip" in rendered:
            return (
                "Thought: The travel planning skill applies.\n"
                "Action: load_skill\n"
                'Action Input: {"skill_name": "travel-planning"}'
            )
        return "Thought: No skill applies.\nFinal Answer: no skill"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8_192


def test_agent_directory_skills_do_not_eagerly_disclose_instructions(
    tmp_path: Path,
) -> None:
    """An agent should initially receive only the skill catalog metadata."""
    _create_skill(
        tmp_path,
        "python-review",
        "Review Python code when the user asks for a code review.",
        "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS",
    )
    _create_skill(
        tmp_path,
        "travel-planning",
        "Plan travel when the user asks for a trip itinerary.",
        "TRAVEL_PRIVATE_INSTRUCTIONS",
    )

    agent = Agent(
        role="Assistant",
        goal="Help with the current request.",
        backstory="A general-purpose assistant.",
        skills=[tmp_path],
    )

    assert agent.skills is not None
    assert [skill.disclosure_level for skill in agent.skills] == [
        METADATA,
        METADATA,
    ]
    assert [skill.instructions for skill in agent.skills] == [None, None]


def test_consecutive_kickoffs_select_skills_without_cross_call_leakage(
    tmp_path: Path,
) -> None:
    """Each execution should independently choose its relevant skill."""
    _create_skill(
        tmp_path,
        "python-review",
        "Review Python code when the user asks for a code review.",
        "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS",
    )
    _create_skill(
        tmp_path,
        "travel-planning",
        "Plan travel when the user asks for a trip itinerary.",
        "TRAVEL_PRIVATE_INSTRUCTIONS",
    )
    agent = Agent(
        role="Assistant",
        goal="Help with the current request.",
        backstory="A general-purpose assistant.",
        skills=[tmp_path],
        llm=_SkillChoosingLLM(model="skill-test"),
        max_iter=3,
    )

    review_output = agent.kickoff("Review this Python change")
    travel_output = agent.kickoff("Plan a weekend trip")
    repeated_review_output = agent.kickoff("Review this Python change")

    assert review_output.raw == "python loaded"
    assert travel_output.raw == "travel loaded"
    assert repeated_review_output.raw == "python loaded"
    assert agent.skills is not None
    assert [skill.disclosure_level for skill in agent.skills] == [
        METADATA,
        METADATA,
    ]
    assert [skill.instructions for skill in agent.skills] == [None, None]

    prompt = Prompts(
        agent=agent,
        has_tools=False,
        use_system_prompt=True,
    ).task_execution()
    rendered = getattr(prompt, "system", "") or prompt.prompt

    assert "python-review" in rendered
    assert "travel-planning" in rendered
    assert "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS" not in rendered
    assert "TRAVEL_PRIVATE_INSTRUCTIONS" not in rendered


def test_each_execution_can_disclose_only_its_relevant_skill(tmp_path: Path) -> None:
    """Loading one skill must not leak or permanently activate another."""
    _create_skill(
        tmp_path,
        "python-review",
        "Review Python code when the user asks for a code review.",
        "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS",
    )
    _create_skill(
        tmp_path,
        "travel-planning",
        "Plan travel when the user asks for a trip itinerary.",
        "TRAVEL_PRIVATE_INSTRUCTIONS",
    )
    agent = Agent(
        role="Assistant",
        goal="Help with the current request.",
        backstory="A general-purpose assistant.",
        skills=[tmp_path],
    )

    review_task = Task(
        description="Review this Python change.",
        expected_output="Review findings.",
        agent=agent,
    )
    agent.create_agent_executor(task=review_task)
    assert agent.agent_executor is not None
    review_loader = next(
        tool
        for tool in agent.agent_executor.original_tools
        if isinstance(tool, LoadSkillTool)
    )
    review_context = review_loader.run(skill_name="python-review")

    assert "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS" in review_context
    assert "TRAVEL_PRIVATE_INSTRUCTIONS" not in review_context

    travel_task = Task(
        description="Plan a weekend trip.",
        expected_output="A travel itinerary.",
        agent=agent,
    )
    agent.create_agent_executor(task=travel_task)
    assert agent.agent_executor is not None
    travel_loader = next(
        tool
        for tool in agent.agent_executor.original_tools
        if isinstance(tool, LoadSkillTool)
    )
    travel_context = travel_loader.run(skill_name="travel-planning")

    assert "TRAVEL_PRIVATE_INSTRUCTIONS" in travel_context
    assert "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS" not in travel_context
    assert agent.skills is not None
    assert [skill.disclosure_level for skill in agent.skills] == [
        METADATA,
        METADATA,
    ]
    assert [skill.instructions for skill in agent.skills] == [None, None]


def test_loader_stays_reachable_when_a_tool_claims_its_default_name(
    tmp_path: Path,
) -> None:
    """A configured tool named load_skill must not shadow the skill loader."""
    _create_skill(
        tmp_path,
        "python-review",
        "Review Python code when the user asks for a code review.",
        "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS",
    )

    agent = Agent(
        role="Assistant",
        goal="Help with the current request.",
        backstory="A general-purpose assistant.",
        skills=[tmp_path],
        tools=[_ConflictingTool()],
    )
    agent.create_agent_executor(
        task=Task(
            description="Review this Python change.",
            expected_output="Review findings.",
            agent=agent,
        )
    )

    assert agent.agent_executor is not None
    tools = agent.agent_executor.original_tools
    loader = next(tool for tool in tools if isinstance(tool, LoadSkillTool))

    assert loader.name != "load_skill"
    assert [tool.name for tool in tools].count(loader.name) == 1
    assert "PYTHON_REVIEW_PRIVATE_INSTRUCTIONS" in loader.run(
        skill_name="python-review"
    )

    prompt = agent.agent_executor.prompt
    rendered = prompt.get("system") or prompt.get("prompt")
    assert f"call `{loader.name}`" in rendered


def test_same_named_skills_from_different_orgs_stay_individually_loadable(
    tmp_path: Path,
) -> None:
    """Skills sharing a frontmatter name must each be addressable."""
    _create_skill(
        tmp_path / "acme",
        "code-review",
        "Review code the Acme way.",
        "ACME_PRIVATE_INSTRUCTIONS",
    )
    _create_skill(
        tmp_path / "globex",
        "code-review",
        "Review code the Globex way.",
        "GLOBEX_PRIVATE_INSTRUCTIONS",
    )

    loader = create_skill_loader_tool(
        [
            load_skill_metadata(tmp_path / "acme" / "code-review"),
            load_skill_metadata(tmp_path / "globex" / "code-review"),
        ]
    )

    assert loader is not None
    assert sorted(loader.catalog) == ["acme/code-review", "globex/code-review"]
    assert "ACME_PRIVATE_INSTRUCTIONS" in loader.run(skill_name="acme/code-review")
    assert "GLOBEX_PRIVATE_INSTRUCTIONS" in loader.run(skill_name="globex/code-review")


def test_loaded_block_and_event_keep_the_catalog_label(tmp_path: Path) -> None:
    """A disambiguated skill must report the label the model was told to load."""
    _create_skill(
        tmp_path / "acme",
        "code-review",
        "Review code the Acme way.",
        "ACME_PRIVATE_INSTRUCTIONS",
    )
    _create_skill(
        tmp_path / "globex",
        "code-review",
        "Review code the Globex way.",
        "GLOBEX_PRIVATE_INSTRUCTIONS",
    )

    loader = create_skill_loader_tool(
        [
            load_skill_metadata(tmp_path / "acme" / "code-review"),
            load_skill_metadata(tmp_path / "globex" / "code-review"),
        ]
    )
    assert loader is not None

    used: list[str] = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(SkillUsedEvent)
        def _record(_source: Any, event: SkillUsedEvent) -> None:
            used.append(event.skill_name)

        context = loader.run(skill_name="globex/code-review")
        assert crewai_event_bus.flush(timeout=10)

    assert '<skill name="globex/code-review">' in context
    assert used == ["globex/code-review"]

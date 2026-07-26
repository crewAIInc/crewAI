"""Regression tests for runtime skill progressive disclosure.

Run this focused test file with:

    uv run pytest lib/crewai/tests/skills/test_progressive_disclosure.py -q
"""

from pathlib import Path
from typing import Any

from crewai import Agent, Task
from crewai.llms.base_llm import BaseLLM
from crewai.skills.models import METADATA
from crewai.skills.tool import LoadSkillTool
from crewai.utilities.prompts import Prompts


def _create_skill(
    parent: Path,
    name: str,
    description: str,
    instructions: str,
) -> None:
    skill_dir = parent / name
    skill_dir.mkdir()
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

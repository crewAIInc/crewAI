# ruff: noqa: T201
"""Manual runner for task-level human feedback.

Usage:
    uv run python scripts/human_feedback_task_runner.py
    uv run python scripts/human_feedback_task_runner.py --auto-feedback
    uv run python scripts/human_feedback_task_runner.py --model openai/gpt-4o-mini

Task-level human review in CrewAI uses ``Task(human_input=True)``. This runner
uses a real configured LLM, so make sure the relevant provider keys are set.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
import os
from unittest.mock import patch

from crewai import Agent, Crew, Process, Task
from crewai.core.providers.human_input import SyncHumanInputProvider


def task_with_human_feedback(
    *,
    name: str,
    description: str,
    expected_output: str,
    agent: Agent,
    human_feedback: bool = True,
) -> Task:
    """Create a task using CrewAI's task-level human review flag."""
    return Task(
        name=name,
        description=description,
        expected_output=expected_output,
        agent=agent,
        human_input=human_feedback,
    )


def build_crew(model: str) -> Crew:
    writer = Agent(
        role="Human Feedback Demo Writer",
        goal="Produce short answers that can be revised after human review",
        backstory="You write compact drafts and respond directly to reviewer feedback.",
        llm=model,
        verbose=True,
    )

    draft = task_with_human_feedback(
        name="draft",
        description="Draft a one-sentence project update for the CrewAI team.",
        expected_output="A concise one-sentence project update.",
        agent=writer,
        human_feedback=True,
    )
    polish = task_with_human_feedback(
        name="polish",
        description="Polish the approved update so it is ready to paste into Slack.",
        expected_output="A final Slack-ready update.",
        agent=writer,
        human_feedback=True,
    )

    return Crew(
        agents=[writer],
        tasks=[draft, polish],
        process=Process.sequential,
        verbose=True,
    )


def auto_feedback_responses() -> Iterator[str]:
    while True:
        yield "Revise it to be warmer and more specific."
        yield ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("CREWAI_HUMAN_FEEDBACK_RUNNER_MODEL", "openai/gpt-4o-mini"),
        help="Real model to use for the runner.",
    )
    parser.add_argument(
        "--auto-feedback",
        action="store_true",
        help="Automatically request one revision per task, then approve it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crew = build_crew(args.model)

    context = nullcontext()
    if args.auto_feedback:
        responses = auto_feedback_responses()
        context = patch.object(
            SyncHumanInputProvider,
            "_prompt_input",
            side_effect=lambda *_args, **_kwargs: next(responses),
        )

    with context:
        result = crew.kickoff()

    print("\n== Final Crew Output ==")
    print(result.raw)


if __name__ == "__main__":
    main()

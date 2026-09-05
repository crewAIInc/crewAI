"""Tests for crewai.agent.utils prompt-building helpers."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from crewai import Task
from crewai.agent.utils import build_task_prompt_with_schema


class _Output(BaseModel):
    name: str
    note: str | None = Field(default=None)


def test_optional_fields_stay_nullable_in_the_prompt_schema() -> None:
    """An Optional field must still be expressible as null in the prompt schema.

    The provider-side response schema generated from the same model allows null,
    so stripping it here hands the model two contradictory contracts and leaves
    it no way to say "not applicable".
    """
    task = Task(description="d", expected_output="e", output_pydantic=_Output)

    prompt = build_task_prompt_with_schema(task, "")

    start = prompt.index("{")
    end = prompt.rindex("}", start) + 1
    schema = json.loads(prompt[start:end])

    assert schema["properties"]["note"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]

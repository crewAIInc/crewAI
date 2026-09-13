"""Tests for crewai.agent.utils prompt-building helpers."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from crewai import Task
from crewai.agent.utils import build_task_prompt_with_schema


class _Output(BaseModel):
    name: str
    note: str | None = Field(default=None)


@pytest.mark.parametrize("output_attribute", ["output_pydantic", "output_json"])
def test_optional_fields_stay_nullable_in_the_prompt_schema(
    output_attribute: str,
) -> None:
    """An Optional field must still be expressible as null in the prompt schema.

    The provider-side response schema generated from the same model allows null,
    so stripping it here hands the model two contradictory contracts and leaves
    it no way to say "not applicable". Both output attributes embed a schema in
    the prompt, so both are pinned.
    """
    task = Task(description="d", expected_output="e", **{output_attribute: _Output})

    prompt = build_task_prompt_with_schema(task, "")

    start = prompt.index("{")
    end = prompt.rindex("}", start) + 1
    schema = json.loads(prompt[start:end])

    assert {entry["type"] for entry in schema["properties"]["note"]["anyOf"]} == {
        "string",
        "null",
    }

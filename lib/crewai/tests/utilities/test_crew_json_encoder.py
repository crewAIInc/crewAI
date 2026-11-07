import json
from pathlib import Path
from typing import Any

from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from crewai.task import Task
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from pydantic import BaseModel, RootModel
import pytest


class Entry(BaseModel):
    value: int


@pytest.fixture(
    params=[
        (RootModel[list[Entry]]([Entry(value=1)]), [{"value": 1}]),
        (RootModel[list[int]]([]), []),
        (RootModel[str]("result"), "result"),
        (RootModel[int](0), 0),
        (RootModel[bool](False), False),
        (RootModel[None](None), None),
        (Entry(value=1), {"value": 1}),
    ],
    ids=["models", "empty-list", "string", "zero", "false", "null", "regular-model"],
)
def model_case(request: pytest.FixtureRequest) -> tuple[BaseModel, Any]:
    """Supply root model shapes and an ordinary model control."""
    return request.param


def test_encode_pydantic_root_values(model_case: tuple[BaseModel, Any]) -> None:
    """Preserve the JSON shape produced by each Pydantic model."""
    model, expected = model_case

    encoded = json.dumps({"pydantic": model}, cls=CrewJSONEncoder)

    assert json.loads(encoded) == {"pydantic": expected}


def test_task_output_storage_round_trips_root_values(
    tmp_path: Path, model_case: tuple[BaseModel, Any]
) -> None:
    """Store and replay outputs containing non-dictionary root models."""
    model, expected = model_case
    task = Task(
        description="Produce structured data", expected_output="Structured data"
    )
    storage = KickoffTaskOutputsSQLiteStorage(str(tmp_path / "outputs.db"))

    storage.add(task, {"pydantic": model}, task_index=0)

    assert storage.load()[0]["output"] == {"pydantic": expected}

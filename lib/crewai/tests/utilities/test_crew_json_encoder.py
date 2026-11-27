from enum import Enum
import json

from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from pydantic import BaseModel


class Status(Enum):
    DONE = "done"


class Result(BaseModel):
    status: Status


def test_enum_serializes_value():
    assert json.dumps(Status.DONE, cls=CrewJSONEncoder) == '"done"'


def test_pydantic_model_with_enum_round_trips():
    result = Result(status=Status.DONE)

    encoded = json.dumps(result, cls=CrewJSONEncoder)

    assert json.loads(encoded) == {"status": "done"}
    assert Result.model_validate_json(encoded) == result

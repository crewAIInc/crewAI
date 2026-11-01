"""Data-only typed dispatch persistence regression tests."""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field
import pytest

from crewai.flow.async_feedback._payloads import decode_payload, encode_payload, payload_types
from crewai.flow.human_feedback import HumanFeedbackResult


class Payload(BaseModel):
    value: int = Field(alias="external_value")


@dataclass
class Envelope:
    payload: Payload


class Outcome(Enum):
    READY = "ready"


def consume(value: Payload, envelope: Envelope, outcome: Outcome) -> None:
    pass


def roundtrip(value):
    types = payload_types([consume], (HumanFeedbackResult,))
    return decode_payload(json.loads(json.dumps(encode_payload(value, types))), types)


@pytest.mark.parametrize("value", [None, True, 42, 0.5, "text", [1, {"a": 2}], (1, "a"), {1, 2}, frozenset({1, 2}), {"kind": "typed", "type": "user-value", "data": 42}, {1: "int-key"}, datetime(2026, 1, 1, tzinfo=timezone.utc), Decimal("1.25"), UUID(int=42), Path("relative/path"), Outcome.READY])
def test_supported_payloads_keep_value_and_type(value):
    restored = roundtrip(value)
    assert type(restored) is type(value)
    assert restored == value


def test_pydantic_aliases_and_dataclass_payloads_keep_runtime_type():
    value = Envelope(Payload(external_value=42))
    restored = roundtrip(value)
    assert isinstance(restored, Envelope)
    assert isinstance(restored.payload, Payload)
    assert restored.payload.value == 42


def test_feedback_result_keeps_nested_typed_output():
    value = HumanFeedbackResult(output=Payload(external_value=42), feedback="approved", outcome="yes")
    restored = roundtrip(value)
    assert isinstance(restored, HumanFeedbackResult)
    assert isinstance(restored.output, Payload)
    assert restored.output.value == 42


def test_persisted_data_cannot_select_an_arbitrary_import():
    with pytest.raises(ValueError, match="not declared"):
        decode_payload({"kind": "typed", "type": "os:system", "data": {"kind": "scalar", "data": "ignored"}}, {})


def test_undeclared_model_is_rejected_instead_of_silently_flattened():
    with pytest.raises(TypeError, match="annotation"):
        encode_payload(Payload(external_value=42), {})


def test_cycles_are_rejected_instead_of_recursing_forever():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="Cyclic"):
        roundtrip(value)



@pytest.mark.parametrize("hidden_kind", ["private", "excluded"])
def test_private_or_excluded_data_is_not_silently_dropped(hidden_kind):
    from pydantic import PrivateAttr
    if hidden_kind == "private":
        class Hidden(BaseModel):
            value: int
            _private: str = PrivateAttr(default="private-value")
    else:
        class Hidden(BaseModel):
            value: int
            hidden: str = Field(default="hidden-value", exclude=True)
    types = payload_types([], (Hidden,))
    with pytest.raises(TypeError, match="persistence adapter"):
        encode_payload(Hidden(value=1), types)


def test_extra_model_fields_are_not_dropped():
    from pydantic import ConfigDict
    class Extensible(BaseModel):
        model_config = ConfigDict(extra="allow")
        value: int
    types = payload_types([], (Extensible,))
    restored = decode_payload(encode_payload(Extensible(value=1, extra_value=42), types), types)
    assert isinstance(restored, Extensible)
    assert restored.extra_value == 42

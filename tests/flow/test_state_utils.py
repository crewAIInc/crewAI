from datetime import date, datetime
from typing import List
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from crewai.flow import Flow
from crewai.flow.state_utils import export_state, to_serializable, to_string


class Address(BaseModel):
    street: str
    city: str
    country: str


class Person(BaseModel):
    name: str
    age: int
    address: Address
    birthday: date
    skills: List[str]


@pytest.fixture
def mock_flow():
    def create_flow(state):
        flow = Mock(spec=Flow)
        flow._state = state
        return flow

    return create_flow


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"text": "hello world"}, {"text": "hello world"}),
        ({"number": 42}, {"number": 42}),
        ({"decimal": 3.14}, {"decimal": 3.14}),
        ({"flag": True}, {"flag": True}),
        ({"empty": None}, {"empty": None}),
        ({"list": [1, 2, 3]}, {"list": [1, 2, 3]}),
        ({"tuple": (1, 2, 3)}, {"tuple": [1, 2, 3]}),
        ({"set": {1, 2, 3}}, {"set": [1, 2, 3]}),
        ({"nested": [1, [2, 3], {4, 5}]}, {"nested": [1, [2, 3], [4, 5]]}),
    ],
)
def test_basic_serialization(mock_flow, test_input, expected):
    flow = mock_flow(test_input)
    result = export_state(flow)
    assert result == expected


@pytest.mark.parametrize(
    "input_date,expected",
    [
        (date(2024, 1, 1), "2024-01-01"),
        (datetime(2024, 1, 1, 12, 30), "2024-01-01T12:30:00"),
    ],
)
def test_temporal_serialization(mock_flow, input_date, expected):
    flow = mock_flow({"date": input_date})
    result = export_state(flow)
    assert result["date"] == expected


@pytest.mark.parametrize(
    "key,value,expected_key_type",
    [
        (("tuple", "key"), "value", str),
        (None, "value", str),
        (123, "value", str),
        ("normal", "value", str),
    ],
)
def test_dictionary_key_serialization(mock_flow, key, value, expected_key_type):
    flow = mock_flow({key: value})
    result = export_state(flow)
    assert len(result) == 1
    result_key = next(iter(result.keys()))
    assert isinstance(result_key, expected_key_type)
    assert result[result_key] == value


@pytest.mark.parametrize(
    "callable_obj,expected_in_result",
    [
        (lambda x: x * 2, "lambda"),
        (str.upper, "upper"),
    ],
)
def test_callable_serialization(mock_flow, callable_obj, expected_in_result):
    flow = mock_flow({"func": callable_obj})
    result = export_state(flow)
    assert isinstance(result["func"], str)
    assert expected_in_result in result["func"].lower()


def test_pydantic_model_serialization(mock_flow):
    address = Address(street="123 Main St", city="Tech City", country="Pythonia")

    person = Person(
        name="John Doe",
        age=30,
        address=address,
        birthday=date(1994, 1, 1),
        skills=["Python", "Testing"],
    )

    flow = mock_flow(
        {
            "single_model": address,
            "nested_model": person,
            "model_list": [address, address],
            "model_dict": {"home": address},
        }
    )

    result = export_state(flow)
    assert (
        to_string(result)
        == '{"single_model": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, "nested_model": {"name": "John Doe", "age": 30, "address": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, "birthday": "1994-01-01", "skills": ["Python", "Testing"]}, "model_list": [{"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}], "model_dict": {"home": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}}}'
    )


def test_depth_limit(mock_flow):
    """Test max depth handling with a deeply nested structure"""

    def create_nested(depth):
        if depth == 0:
            return "value"
        return {"next": create_nested(depth - 1)}

    deep_structure = create_nested(10)
    flow = mock_flow(deep_structure)
    result = export_state(flow)

    assert result == {
        "next": {
            "next": {
                "next": {
                    "next": {
                        "next": "{'next': {'next': {'next': {'next': {'next': 'value'}}}}}"
                    }
                }
            }
        }
    }


def test_exclude_keys():
    result = to_serializable({"key1": "value1", "key2": "value2"}, exclude={"key1"})
    assert result == {"key2": "value2"}

    model = Person(
        name="John Doe",
        age=30,
        address=Address(street="123 Main St", city="Tech City", country="Pythonia"),
        birthday=date(1994, 1, 1),
        skills=["Python", "Testing"],
    )
    result = to_serializable(model, exclude={"address"})
    assert result == {
        "name": "John Doe",
        "age": 30,
        "birthday": "1994-01-01",
        "skills": ["Python", "Testing"],
    }

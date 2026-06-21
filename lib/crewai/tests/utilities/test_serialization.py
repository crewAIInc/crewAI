from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, List

import pytest
from crewai.utilities.serialization import to_serializable, to_string
from pydantic import BaseModel


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


class Container(BaseModel):
    payload: BaseModel | None = None


@dataclass
class DataclassPerson:
    name: str
    address: Address
    skills: tuple[str, ...]


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
def test_basic_serialization(test_input, expected):
    result = to_serializable(test_input)
    assert result == expected


@pytest.mark.parametrize(
    "input_date,expected",
    [
        (date(2024, 1, 1), "2024-01-01"),
        (datetime(2024, 1, 1, 12, 30), "2024-01-01T12:30:00"),
    ],
)
def test_temporal_serialization(input_date, expected):
    result = to_serializable({"date": input_date})
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
def test_dictionary_key_serialization(key, value, expected_key_type):
    result = to_serializable({key: value})
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
def test_callable_serialization(callable_obj, expected_in_result):
    result = to_serializable({"func": callable_obj})
    assert isinstance(result["func"], str)
    assert expected_in_result in result["func"].lower()


def test_pydantic_model_serialization():
    address = Address(street="123 Main St", city="Tech City", country="Pythonia")

    person = Person(
        name="John Doe",
        age=30,
        address=address,
        birthday=date(1994, 1, 1),
        skills=["Python", "Testing"],
    )

    data = {
        "single_model": address,
        "nested_model": person,
        "model_list": [address, address],
        "model_dict": {"home": address},
    }

    result = to_serializable(data)
    assert (
        to_string(result)
        == '{"single_model": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, "nested_model": {"name": "John Doe", "age": 30, "address": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, "birthday": "1994-01-01", "skills": ["Python", "Testing"]}, "model_list": [{"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}, {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}], "model_dict": {"home": {"street": "123 Main St", "city": "Tech City", "country": "Pythonia"}}}'
    )


def test_polymorphic_field_serializes_concrete_subclass():
    container = Container(
        payload=Address(street="1 Main", city="Tech City", country="Pythonia")
    )

    assert to_serializable(container) == {
        "payload": {"street": "1 Main", "city": "Tech City", "country": "Pythonia"}
    }


def test_dataclass_serialization_recurses_into_nested_values():
    person = DataclassPerson(
        name="Ada",
        address=Address(street="1 Loop", city="Compute", country="Pythonia"),
        skills=("Python", "Math"),
    )

    assert to_serializable(person) == {
        "name": "Ada",
        "address": {
            "street": "1 Loop",
            "city": "Compute",
            "country": "Pythonia",
        },
        "skills": ["Python", "Math"],
    }


def test_depth_limit():
    """Test max depth handling with a deeply nested structure"""

    def create_nested(depth):
        if depth == 0:
            return "value"
        return {"next": create_nested(depth - 1)}

    deep_structure = create_nested(10)
    result = to_serializable(deep_structure)

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


@pytest.mark.parametrize("max_depth", [0, -1])
def test_non_positive_max_depth_disables_depth_limit(max_depth):
    def create_nested(depth):
        if depth == 0:
            return "value"
        return {"next": create_nested(depth - 1)}

    assert to_serializable(create_nested(10), max_depth=max_depth) == create_nested(10)


def test_unlimited_depth_still_detects_dataclass_cycles():
    @dataclass
    class Node:
        child: Any = None

    node = Node()
    node.child = node

    assert to_serializable(node, max_depth=0) == {"child": "<circular_ref:Node>"}


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

import json
import os
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from crewai.flow.state_utils import _to_serializable_key, to_serializable, to_string


# Sample Pydantic models for testing
class SimpleModel(BaseModel):
    name: str
    age: int


class NestedModel(BaseModel):
    id: int
    data: SimpleModel


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class EnumModel(BaseModel):
    name: str
    color: Color


class OptionalModel(BaseModel):
    name: str
    age: Optional[int]


class ListModel(BaseModel):
    items: List[int]


class UnionModel(BaseModel):
    field: Union[int, str, None]


# Tests for to_serializable function
def test_to_serializable_primitives():
    """Test serialization of primitive types."""
    assert to_serializable("test string") == "test string"
    assert to_serializable(42) == 42
    assert to_serializable(3.14) == 3.14
    assert to_serializable(True) == True
    assert to_serializable(None) is None


def test_to_serializable_dates():
    """Test serialization of date and datetime objects."""
    test_date = date(2023, 1, 15)
    test_datetime = datetime(2023, 1, 15, 10, 30, 45)

    assert to_serializable(test_date) == "2023-01-15"
    assert to_serializable(test_datetime) == "2023-01-15T10:30:45"


def test_to_serializable_collections():
    """Test serialization of lists, tuples, and sets."""
    test_list = [1, "two", 3.0]
    test_tuple = (4, "five", 6.0)
    test_set = {7, "eight", 9.0}

    assert to_serializable(test_list) == [1, "two", 3.0]
    assert to_serializable(test_tuple) == [4, "five", 6.0]

    # For sets, we can't rely on order, so we'll verify differently
    serialized_set = to_serializable(test_set)
    assert isinstance(serialized_set, list)
    assert len(serialized_set) == 3
    assert 7 in serialized_set
    assert "eight" in serialized_set
    assert 9.0 in serialized_set


def test_to_serializable_dict():
    """Test serialization of dictionaries."""
    test_dict = {"a": 1, "b": "two", "c": [3, 4, 5]}

    assert to_serializable(test_dict) == {"a": 1, "b": "two", "c": [3, 4, 5]}


def test_to_serializable_pydantic_models():
    """Test serialization of Pydantic models."""
    simple = SimpleModel(name="John", age=30)

    assert to_serializable(simple) == {"name": "John", "age": 30}


def test_to_serializable_nested_models():
    """Test serialization of nested Pydantic models."""
    simple = SimpleModel(name="John", age=30)
    nested = NestedModel(id=1, data=simple)

    assert to_serializable(nested) == {"id": 1, "data": {"name": "John", "age": 30}}


def test_to_serializable_complex_model():
    """Test serialization of a complex model with nested structures."""
    person = Person(
        name="Jane",
        age=28,
        address=Address(street="123 Main St", city="Anytown", zip_code="12345"),
    )

    assert to_serializable(person) == {
        "name": "Jane",
        "age": 28,
        "address": {"street": "123 Main St", "city": "Anytown", "zip_code": "12345"},
    }


def test_to_serializable_enum():
    """Test serialization of Enum values."""
    model = EnumModel(name="ColorTest", color=Color.RED)

    assert to_serializable(model) == {"name": "ColorTest", "color": "red"}


def test_to_serializable_optional_fields():
    """Test serialization of models with optional fields."""
    model_with_age = OptionalModel(name="WithAge", age=25)
    model_without_age = OptionalModel(name="WithoutAge", age=None)

    assert to_serializable(model_with_age) == {"name": "WithAge", "age": 25}
    assert to_serializable(model_without_age) == {"name": "WithoutAge", "age": None}


def test_to_serializable_list_field():
    """Test serialization of models with list fields."""
    model = ListModel(items=[1, 2, 3, 4, 5])

    assert to_serializable(model) == {"items": [1, 2, 3, 4, 5]}


def test_to_serializable_union_field():
    """Test serialization of models with union fields."""
    model_int = UnionModel(field=42)
    model_str = UnionModel(field="test")
    model_none = UnionModel(field=None)

    assert to_serializable(model_int) == {"field": 42}
    assert to_serializable(model_str) == {"field": "test"}
    assert to_serializable(model_none) == {"field": None}


def test_to_serializable_max_depth():
    """Test max depth parameter to prevent infinite recursion."""
    # Create recursive structure
    a: Dict[str, Any] = {"name": "a"}
    b: Dict[str, Any] = {"name": "b", "ref": a}
    a["ref"] = b  # Create circular reference

    result = to_serializable(a, max_depth=3)

    assert isinstance(result, dict)
    assert "name" in result
    assert "ref" in result
    assert isinstance(result["ref"], dict)
    assert "ref" in result["ref"]
    assert isinstance(result["ref"]["ref"], dict)
    # At depth 3, it should convert to string
    assert isinstance(result["ref"]["ref"]["ref"], str)


def test_to_serializable_non_serializable():
    """Test serialization of objects that aren't directly JSON serializable."""

    class CustomObject:
        def __repr__(self):
            return "CustomObject()"

    obj = CustomObject()

    # Should convert to string representation
    assert to_serializable(obj) == "CustomObject()"


def test_to_string_conversion():
    """Test the to_string function."""
    test_dict = {"name": "Test", "values": [1, 2, 3]}

    # Should convert to a JSON string
    assert to_string(test_dict) == '{"name": "Test", "values": [1, 2, 3]}'

    # None should return None
    assert to_string(None) is None


def test_to_serializable_key():
    """Test serialization of dictionary keys."""
    # String and int keys are converted to strings
    assert _to_serializable_key("test") == "test"
    assert _to_serializable_key(42) == "42"

    # Complex objects are converted to a unique string
    obj = object()
    key_str = _to_serializable_key(obj)
    assert isinstance(key_str, str)
    assert "key_" in key_str
    assert "object" in key_str

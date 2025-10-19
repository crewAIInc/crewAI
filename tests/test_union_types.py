"""Test Union type support in Pydantic outputs."""
import json
from typing import Union
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from crewai.utilities.converter import (
    convert_to_model,
    generate_model_description,
)
from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


class SuccessData(BaseModel):
    """Model for successful response."""
    status: str
    result: str
    value: int


class ErrorMessage(BaseModel):
    """Model for error response."""
    status: str
    error: str
    code: int


class ResponseWithUnion(BaseModel):
    """Model with Union type field."""
    response: Union[SuccessData, ErrorMessage]


class DirectUnionModel(BaseModel):
    """Model with direct Union type."""
    data: Union[str, int, dict]


class MultiUnionModel(BaseModel):
    """Model with multiple Union types."""
    field1: Union[str, int]
    field2: Union[SuccessData, ErrorMessage, None]


def test_convert_to_model_with_union_success_data():
    """Test converting JSON to a model with Union type (SuccessData variant)."""
    result = json.dumps({
        "response": {
            "status": "success",
            "result": "Operation completed",
            "value": 42
        }
    })
    
    output = convert_to_model(result, ResponseWithUnion, None, None)
    assert isinstance(output, ResponseWithUnion)
    assert isinstance(output.response, SuccessData)
    assert output.response.status == "success"
    assert output.response.result == "Operation completed"
    assert output.response.value == 42


def test_convert_to_model_with_union_error_message():
    """Test converting JSON to a model with Union type (ErrorMessage variant)."""
    result = json.dumps({
        "response": {
            "status": "error",
            "error": "Something went wrong",
            "code": 500
        }
    })
    
    output = convert_to_model(result, ResponseWithUnion, None, None)
    assert isinstance(output, ResponseWithUnion)
    assert isinstance(output.response, ErrorMessage)
    assert output.response.status == "error"
    assert output.response.error == "Something went wrong"
    assert output.response.code == 500


def test_convert_to_model_with_direct_union_string():
    """Test converting JSON to a model with direct Union type (string variant)."""
    result = json.dumps({"data": "hello world"})
    
    output = convert_to_model(result, DirectUnionModel, None, None)
    assert isinstance(output, DirectUnionModel)
    assert isinstance(output.data, str)
    assert output.data == "hello world"


def test_convert_to_model_with_direct_union_int():
    """Test converting JSON to a model with direct Union type (int variant)."""
    result = json.dumps({"data": 42})
    
    output = convert_to_model(result, DirectUnionModel, None, None)
    assert isinstance(output, DirectUnionModel)
    assert isinstance(output.data, int)
    assert output.data == 42


def test_convert_to_model_with_direct_union_dict():
    """Test converting JSON to a model with direct Union type (dict variant)."""
    result = json.dumps({"data": {"key": "value", "number": 123}})
    
    output = convert_to_model(result, DirectUnionModel, None, None)
    assert isinstance(output, DirectUnionModel)
    assert isinstance(output.data, dict)
    assert output.data == {"key": "value", "number": 123}


def test_convert_to_model_with_multiple_unions():
    """Test converting JSON to a model with multiple Union type fields."""
    result = json.dumps({
        "field1": "text",
        "field2": {
            "status": "success",
            "result": "Done",
            "value": 100
        }
    })
    
    output = convert_to_model(result, MultiUnionModel, None, None)
    assert isinstance(output, MultiUnionModel)
    assert isinstance(output.field1, str)
    assert output.field1 == "text"
    assert isinstance(output.field2, SuccessData)
    assert output.field2.status == "success"


def test_convert_to_model_with_optional_union_none():
    """Test converting JSON to a model with optional Union type (None variant)."""
    result = json.dumps({
        "field1": 42,
        "field2": None
    })
    
    output = convert_to_model(result, MultiUnionModel, None, None)
    assert isinstance(output, MultiUnionModel)
    assert isinstance(output.field1, int)
    assert output.field1 == 42
    assert output.field2 is None


def test_generate_model_description_with_union():
    """Test that generate_model_description handles Union types correctly."""
    description = generate_model_description(ResponseWithUnion)
    
    assert "Union" in description
    assert "Optional" not in description
    assert "status" in description
    print(f"Generated description:\n{description}")


def test_generate_model_description_with_direct_union():
    """Test that generate_model_description handles direct Union types correctly."""
    description = generate_model_description(DirectUnionModel)
    
    assert "Union" in description
    assert "Optional" not in description
    assert "str" in description and "int" in description and "dict" in description
    print(f"Generated description:\n{description}")


def test_pydantic_schema_parser_with_union():
    """Test that PydanticSchemaParser handles Union types correctly."""
    parser = PydanticSchemaParser(model=ResponseWithUnion)
    schema = parser.get_schema()
    
    assert "Union" in schema or "SuccessData" in schema or "ErrorMessage" in schema
    print(f"Generated schema:\n{schema}")


def test_pydantic_schema_parser_with_direct_union():
    """Test that PydanticSchemaParser handles direct Union types correctly."""
    parser = PydanticSchemaParser(model=DirectUnionModel)
    schema = parser.get_schema()
    
    assert "Union" in schema or ("str" in schema and "int" in schema and "dict" in schema)
    print(f"Generated schema:\n{schema}")


def test_pydantic_schema_parser_with_optional_union():
    """Test that PydanticSchemaParser handles Optional Union types correctly."""
    parser = PydanticSchemaParser(model=MultiUnionModel)
    schema = parser.get_schema()
    
    assert "Union" in schema or "Optional" in schema
    print(f"Generated schema:\n{schema}")


def test_generate_model_description_with_optional_union():
    """Test that generate_model_description correctly wraps Optional Union types."""
    description = generate_model_description(MultiUnionModel)
    
    assert "field1" in description
    assert "field2" in description
    assert "Optional" in description
    print(f"Generated description:\n{description}")

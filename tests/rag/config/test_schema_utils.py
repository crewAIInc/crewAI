"""Tests for RAG config schema utilities."""

import pytest
from dataclasses import dataclass, field
from typing import Literal
from pydantic import BaseModel, GetCoreSchemaHandler

from crewai.rag.config.schema_utils import (
    serialize_pydantic_settings,
    create_dataclass_schema,
)


class MockSettings(BaseModel):
    """Mock Pydantic settings for testing."""

    api_key: str = "test_key"
    timeout: int = 30


class OldMockSettings:
    """Mock settings with deprecated dict method."""

    def dict(self):
        return {"api_key": "old_key", "timeout": 10}


def test_serialize_pydantic_settings_with_model_dump():
    """Test serialization with model_dump method."""
    settings = MockSettings()
    result = serialize_pydantic_settings(settings)
    assert result == {"api_key": "test_key", "timeout": 30}


def test_serialize_pydantic_settings_with_dict_method():
    """Test serialization with deprecated dict method."""
    settings = OldMockSettings()
    result = serialize_pydantic_settings(settings)
    assert result == {"api_key": "old_key", "timeout": 10}


def test_serialize_pydantic_settings_raises_error():
    """Test serialization raises error when neither method exists."""

    class BadSettings:
        pass

    with pytest.raises(
        ValueError, match="Settings object lacks 'dict' or 'model_dump' method"
    ):
        serialize_pydantic_settings(BadSettings())


def test_create_dataclass_schema():
    """Test dataclass schema creation returns proper structure."""

    @dataclass
    class TestConfig:
        provider: Literal["test"] = "test"
        enabled: bool = True
        count: int | None = None
        settings: MockSettings | None = field(default=None)

    from unittest.mock import Mock

    handler = Mock(spec=GetCoreSchemaHandler)
    schema = create_dataclass_schema(TestConfig, handler)

    assert schema["type"] == "dataclass"
    assert "schema" in schema
    assert "fields" in schema

    field_names = schema["fields"]
    assert "provider" in field_names
    assert "enabled" in field_names
    assert "count" in field_names
    assert "settings" in field_names

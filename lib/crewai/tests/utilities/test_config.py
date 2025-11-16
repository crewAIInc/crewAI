"""Tests for utilities.config.process_config function."""

import pytest
from pydantic import BaseModel, Field

from crewai.utilities.config import process_config
from crewai.utilities.constants import NOT_SPECIFIED


class TestProcessConfig:
    """Test suite for process_config function."""

    def test_process_config_with_none_overrides_not_specified(self):
        """Test that config with None value overrides NOT_SPECIFIED sentinel."""
        
        class TestModel(BaseModel):
            context: list[str] | None | type(NOT_SPECIFIED) = Field(default=NOT_SPECIFIED)
            description: str = "default"

        values = {
            "context": NOT_SPECIFIED,
            "description": "test",
            "config": {"context": None}
        }

        result = process_config(values, TestModel)

        assert result["context"] is None
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_with_none_overrides_none(self):
        """Test that config with None value can override existing None."""
        
        class TestModel(BaseModel):
            context: list[str] | None = None
            description: str = "default"

        values = {
            "context": None,
            "description": "test",
            "config": {"context": None}
        }

        result = process_config(values, TestModel)

        assert result["context"] is None
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_preserves_explicit_values(self):
        """Test that config does not override explicitly set non-None values."""
        
        class TestModel(BaseModel):
            context: list[str] | None = None
            description: str = "default"

        values = {
            "context": ["task1", "task2"],
            "description": "test",
            "config": {"context": None}
        }

        result = process_config(values, TestModel)

        assert result["context"] == ["task1", "task2"]
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_with_empty_list_from_config(self):
        """Test that config with empty list is preserved."""
        
        class TestModel(BaseModel):
            context: list[str] | None | type(NOT_SPECIFIED) = Field(default=NOT_SPECIFIED)
            description: str = "default"

        values = {
            "context": NOT_SPECIFIED,
            "description": "test",
            "config": {"context": []}
        }

        result = process_config(values, TestModel)

        assert result["context"] == []
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_does_not_override_false(self):
        """Test that config does not override explicit False value."""
        
        class TestModel(BaseModel):
            flag: bool = True
            description: str = "default"

        values = {
            "flag": False,
            "description": "test",
            "config": {"flag": True}
        }

        result = process_config(values, TestModel)

        assert result["flag"] is False
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_does_not_override_zero(self):
        """Test that config does not override explicit 0 value."""
        
        class TestModel(BaseModel):
            count: int = 10
            description: str = "default"

        values = {
            "count": 0,
            "description": "test",
            "config": {"count": 5}
        }

        result = process_config(values, TestModel)

        assert result["count"] == 0
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_does_not_override_empty_string(self):
        """Test that config does not override explicit empty string value."""
        
        class TestModel(BaseModel):
            name: str = "default"
            description: str = "default"

        values = {
            "name": "",
            "description": "test",
            "config": {"name": "new_name"}
        }

        result = process_config(values, TestModel)

        assert result["name"] == ""
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_with_dict_merge(self):
        """Test that config properly merges dict values."""
        
        class TestModel(BaseModel):
            settings: dict[str, str] = Field(default_factory=dict)
            description: str = "default"

        values = {
            "settings": {"key1": "value1"},
            "description": "test",
            "config": {"settings": {"key2": "value2"}}
        }

        result = process_config(values, TestModel)

        assert result["settings"] == {"key1": "value1", "key2": "value2"}
        assert result["description"] == "test"
        assert "config" not in result

    def test_process_config_with_no_config(self):
        """Test that process_config handles missing config gracefully."""
        
        class TestModel(BaseModel):
            context: list[str] | None = None
            description: str = "default"

        values = {
            "context": None,
            "description": "test"
        }

        result = process_config(values, TestModel)

        assert result["context"] is None
        assert result["description"] == "test"

    def test_process_config_with_empty_config(self):
        """Test that process_config handles empty config gracefully."""
        
        class TestModel(BaseModel):
            context: list[str] | None = None
            description: str = "default"

        values = {
            "context": None,
            "description": "test",
            "config": {}
        }

        result = process_config(values, TestModel)

        assert result["context"] is None
        assert result["description"] == "test"
        assert "config" not in result


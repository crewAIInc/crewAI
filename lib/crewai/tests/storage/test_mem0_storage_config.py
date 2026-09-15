"""Tests for Mem0Storage._parse_config JSON string handling.

Verifies the fix for https://github.com/crewAIInc/crewAI/issues/4423
"""

import json

import pytest

from crewai.memory.storage.mem0_storage import Mem0Storage


class TestParseConfig:
    """Unit tests for Mem0Storage._parse_config (static method)."""

    def test_none_returns_empty_dict(self):
        assert Mem0Storage._parse_config(None) == {}

    def test_dict_passes_through(self):
        cfg = {"user_id": "u1", "agent_id": "a1"}
        assert Mem0Storage._parse_config(cfg) is cfg

    def test_json_string_parsed(self):
        raw = '{"user_id": "u1", "agent_id": "a1"}'
        result = Mem0Storage._parse_config(raw)
        assert result == {"user_id": "u1", "agent_id": "a1"}
        assert isinstance(result, dict)

    def test_nested_json_string_parsed(self):
        raw = json.dumps({
            "user_id": "u1",
            "local_mem0_config": {"vector_store": {"provider": "qdrant"}},
        })
        result = Mem0Storage._parse_config(raw)
        assert result["local_mem0_config"]["vector_store"]["provider"] == "qdrant"

    def test_invalid_json_string_raises_value_error(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            Mem0Storage._parse_config("{bad json}")

    def test_json_array_raises_type_error(self):
        with pytest.raises(TypeError, match="must decode to a dict"):
            Mem0Storage._parse_config('[1, 2, 3]')

    def test_json_scalar_raises_type_error(self):
        with pytest.raises(TypeError, match="must decode to a dict"):
            Mem0Storage._parse_config('"just a string"')

    def test_integer_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            Mem0Storage._parse_config(42)

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            Mem0Storage._parse_config([1, 2])

    def test_empty_json_object_string(self):
        assert Mem0Storage._parse_config("{}") == {}

    def test_empty_dict(self):
        assert Mem0Storage._parse_config({}) == {}

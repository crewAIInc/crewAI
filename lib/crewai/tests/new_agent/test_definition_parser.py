"""Tests for the agent definition parser and JSON Schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from crewai.new_agent.definition_parser import (
    load_agent_from_definition,
    parse_agent_definition,
    strip_jsonc_comments,
)


class TestStripJsoncComments:
    def test_no_comments(self):
        text = '{"key": "value"}'
        assert json.loads(strip_jsonc_comments(text)) == {"key": "value"}

    def test_single_line_comments(self):
        text = '{\n  // This is a comment\n  "key": "value"\n}'
        result = json.loads(strip_jsonc_comments(text))
        assert result == {"key": "value"}

    def test_multi_line_comments(self):
        text = '{\n  /* This is\n  a multi-line comment */\n  "key": "value"\n}'
        result = json.loads(strip_jsonc_comments(text))
        assert result == {"key": "value"}

    def test_url_in_value_not_stripped(self):
        text = '{"url": "https://example.com"}'
        result = json.loads(strip_jsonc_comments(text))
        assert result["url"] == "https://example.com"


class TestParseAgentDefinition:
    def test_parse_dict(self):
        defn = {"role": "R", "goal": "g"}
        result = parse_agent_definition(defn)
        assert result == defn

    def test_parse_json_string(self):
        raw = '{"role": "R", "goal": "g"}'
        result = parse_agent_definition(raw)
        assert result["role"] == "R"

    def test_parse_json_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"role": "Writer", "goal": "Write articles"}, f)
            f.flush()
            result = parse_agent_definition(f.name)
        assert result["role"] == "Writer"

    def test_parse_jsonc_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonc", mode="w", delete=False) as f:
            f.write('{\n  // Agent definition\n  "role": "Writer",\n  "goal": "Write"\n}')
            f.flush()
            result = parse_agent_definition(f.name)
        assert result["role"] == "Writer"


class TestLoadAgentFromDefinition:
    def test_basic_definition(self):
        defn = {
            "role": "Senior Researcher",
            "goal": "Find information",
            "backstory": "Expert researcher.",
        }
        agent = load_agent_from_definition(defn)
        assert agent.role == "Senior Researcher"
        assert agent.goal == "Find information"
        assert agent.backstory == "Expert researcher."

    def test_minimal_definition(self):
        agent = load_agent_from_definition({"role": "R", "goal": "g"})
        assert agent.role == "R"
        assert agent.goal == "g"

    def test_settings_mapping(self):
        defn = {
            "role": "R",
            "goal": "g",
            "settings": {
                "memory": False,
                "reasoning": False,
                "planning": False,
                "narration_guard": True,
                "max_history_messages": 50,
            },
        }
        agent = load_agent_from_definition(defn)
        assert agent.settings.memory_enabled is False
        assert agent.settings.reasoning_enabled is False
        assert agent.settings.planning_enabled is False
        assert agent.settings.narration_guard is True
        assert agent.settings.max_history_messages == 50

    def test_verbose_and_max_iter(self):
        defn = {"role": "R", "goal": "g", "verbose": True, "max_iter": 10}
        agent = load_agent_from_definition(defn)
        assert agent.verbose is True
        assert agent.max_iter == 10

    def test_llm_setting(self):
        defn = {"role": "R", "goal": "g", "llm": "openai/gpt-4o"}
        agent = load_agent_from_definition(defn)
        assert agent.llm == "openai/gpt-4o"

    def test_guardrail_llm(self):
        defn = {
            "role": "R",
            "goal": "g",
            "guardrail": {"type": "llm", "instructions": "Be safe"},
        }
        agent = load_agent_from_definition(defn)
        assert agent.guardrail is not None
        from crewai.tasks.llm_guardrail import LLMGuardrail
        assert isinstance(agent.guardrail, LLMGuardrail)
        assert agent.guardrail.description == "Be safe"

    def test_from_json_file(self):
        defn = {"role": "FileAgent", "goal": "Test file loading", "backstory": "From JSON"}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(defn, f)
            f.flush()
            agent = load_agent_from_definition(f.name)
        assert agent.role == "FileAgent"
        assert agent.backstory == "From JSON"

    def test_coworker_amp_handle(self):
        defn = {
            "role": "Manager",
            "goal": "Manage",
            "coworkers": [{"amp": "content-writer"}],
        }
        agent = load_agent_from_definition(defn)
        # AMP handles are passed as strings for resolution
        assert "content-writer" in agent.coworkers

    def test_coworker_ref_with_agents_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            writer_defn = {"role": "Writer", "goal": "Write"}
            (agents_dir / "writer.json").write_text(json.dumps(writer_defn))

            defn = {
                "role": "Manager",
                "goal": "Manage",
                "coworkers": [{"ref": "writer"}],
            }
            agent = load_agent_from_definition(defn, agents_dir=agents_dir)
            assert len(agent.coworkers) == 1


    def test_circular_coworker_ref_no_crash(self):
        """Two agents referencing each other as coworkers should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir)
            a_defn = {
                "name": "agent_a",
                "role": "A",
                "goal": "Do A",
                "coworkers": [{"ref": "agent_b"}],
            }
            b_defn = {
                "name": "agent_b",
                "role": "B",
                "goal": "Do B",
                "coworkers": [{"ref": "agent_a"}],
            }
            (agents_dir / "agent_a.json").write_text(json.dumps(a_defn))
            (agents_dir / "agent_b.json").write_text(json.dumps(b_defn))

            agent = load_agent_from_definition(
                agents_dir / "agent_a.json", agents_dir=agents_dir
            )
            assert agent is not None
            assert agent.role == "A"
            # B should be loaded as a coworker, but B's ref to A is skipped
            assert len(agent.coworkers) == 1


class TestJsonSchema:
    def test_schema_is_valid_json(self):
        schema_path = Path(__file__).parent.parent.parent / "src" / "crewai" / "new_agent" / "agent_schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "role" in schema["required"]
        assert "goal" in schema["required"]

    def test_schema_has_key_properties(self):
        schema_path = Path(__file__).parent.parent.parent / "src" / "crewai" / "new_agent" / "agent_schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        props = schema["properties"]
        assert "role" in props
        assert "goal" in props
        assert "backstory" in props
        assert "llm" in props
        assert "tools" in props
        assert "coworkers" in props
        assert "settings" in props
        assert "guardrail" in props

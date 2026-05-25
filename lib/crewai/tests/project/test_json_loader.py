"""Tests for crewai.project.json_loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crewai.project.json_loader import load_agent, strip_jsonc_comments


class TestStripJsoncComments:
    def test_strips_single_line_comments(self):
        text = '{\n  "key": "value" // this is a comment\n}'
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_strips_block_comments(self):
        text = '{\n  /* block comment */\n  "key": "value"\n}'
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_preserves_urls_with_double_slash(self):
        text = '{\n  "url": "https://example.com"\n}'
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data["url"] == "https://example.com"

    def test_removes_trailing_commas(self):
        text = '{\n  "a": 1,\n  "b": 2,\n}'
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data == {"a": 1, "b": 2}

    def test_removes_trailing_commas_in_arrays(self):
        text = '{"arr": [1, 2, 3,]}'
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data["arr"] == [1, 2, 3]

    def test_plain_json_unchanged(self):
        text = '{"key": "value"}'
        result = strip_jsonc_comments(text)
        assert json.loads(result) == {"key": "value"}


class TestLoadAgent:
    def test_load_minimal_agent(self, tmp_path: Path):
        agent_def = {
            "role": "Researcher",
            "goal": "Find information",
            "backstory": "Expert researcher.",
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.role == "Researcher"
        assert agent.goal == "Find information"
        assert agent.backstory == "Expert researcher."

    def test_load_agent_with_llm(self, tmp_path: Path):
        agent_def = {
            "role": "Coder",
            "goal": "Write code",
            "backstory": "Expert coder.",
            "llm": "openai/gpt-4o",
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.role == "Coder"

    def test_load_agent_with_settings_block(self, tmp_path: Path):
        agent_def = {
            "role": "Analyst",
            "goal": "Analyze data",
            "backstory": "Data expert.",
            "settings": {
                "verbose": True,
                "allow_delegation": True,
                "max_iter": 10,
                "cache": False,
            },
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.role == "Analyst"
        assert agent.verbose is True
        assert agent.allow_delegation is True
        assert agent.max_iter == 10
        assert agent.cache is False

    def test_load_agent_with_top_level_settings(self, tmp_path: Path):
        agent_def = {
            "role": "Analyst",
            "goal": "Analyze data",
            "backstory": "Data expert.",
            "verbose": True,
            "max_iter": 15,
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.verbose is True
        assert agent.max_iter == 15

    def test_settings_block_takes_precedence(self, tmp_path: Path):
        agent_def = {
            "role": "Analyst",
            "goal": "Analyze data",
            "backstory": "Data expert.",
            "verbose": False,
            "settings": {
                "verbose": True,
            },
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.verbose is True

    def test_load_agent_from_jsonc(self, tmp_path: Path):
        jsonc_content = """{
  // This is a JSONC file with comments
  "role": "Writer",
  "goal": "Write articles",
  "backstory": "Expert writer.",
  /* multi-line
     comment */
}"""
        agent_file = tmp_path / "agent.jsonc"
        agent_file.write_text(jsonc_content)

        agent = load_agent(agent_file)
        assert agent.role == "Writer"

    def test_load_agent_missing_required_fields(self, tmp_path: Path):
        agent_def = {"role": "Incomplete"}
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        with pytest.raises(Exception):
            load_agent(agent_file)

    def test_load_agent_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_agent(Path("/nonexistent/agent.json"))

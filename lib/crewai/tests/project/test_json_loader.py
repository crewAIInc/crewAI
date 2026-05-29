"""Tests for crewai.project.json_loader."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from crewai.project.json_loader import (
    JSONProjectValidationError,
    find_json_project_file,
    load_agent,
    strip_jsonc_comments,
)


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

    def test_preserves_comment_markers_inside_strings(self):
        text = """{
  "url": "https://example.com/a//b",
  "pattern": "keep /* this */ text",
  "text": "value // not a comment",
}"""
        result = strip_jsonc_comments(text)
        data = json.loads(result)
        assert data["url"] == "https://example.com/a//b"
        assert data["pattern"] == "keep /* this */ text"
        assert data["text"] == "value // not a comment"

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


def test_find_json_project_file_prefers_jsonc(tmp_path: Path):
    (tmp_path / "agent.json").write_text("{}")
    jsonc_path = tmp_path / "agent.jsonc"
    jsonc_path.write_text("{}")

    assert find_json_project_file(tmp_path, "agent") == jsonc_path


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

    def test_load_agent_accepts_public_agent_config_fields(self, tmp_path: Path):
        agent_def = {
            "role": "Analyst",
            "goal": "Analyze data",
            "backstory": "Data expert.",
            "max_execution_time": 30,
            "use_system_prompt": False,
            "system_template": "system: {{ .System }}",
            "prompt_template": "prompt: {{ .Prompt }}",
            "response_template": "response: {{ .Response }}",
            "inject_date": True,
            "date_format": "%Y",
            "guardrail": "Only return concise answers.",
            "guardrail_max_retries": 1,
            "security_config": {"fingerprint": "agent-seed"},
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert agent.max_execution_time == 30
        assert agent.use_system_prompt is False
        assert agent.system_template == "system: {{ .System }}"
        assert agent.inject_date is True
        assert agent.guardrail == "Only return concise answers."

    def test_load_agent_accepts_serialized_tool_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        module = tmp_path / "test_tools.py"
        module.write_text(
            "from crewai.tools.base_tool import BaseTool\n"
            "class EchoTool(BaseTool):\n"
            "    name: str = 'echo'\n"
            "    description: str = 'Echo input'\n"
            "    def _run(self, value: str = '') -> str:\n"
            "        return value\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("test_tools", None)

        agent_def = {
            "role": "Tool User",
            "goal": "Use tools",
            "backstory": "Tool expert.",
            "tools": [
                {
                    "tool_type": "test_tools.EchoTool",
                    "name": "echo",
                    "description": "Echo input",
                }
            ],
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)
        assert len(agent.tools or []) == 1
        assert agent.tools[0].name == "echo"

    def test_load_agent_rejects_runtime_fields(self, tmp_path: Path):
        agent_def = {
            "id": "00000000-0000-4000-8000-000000000000",
            "role": "Analyst",
            "goal": "Analyze data",
            "backstory": "Data expert.",
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        with pytest.raises(JSONProjectValidationError, match="runtime-only"):
            load_agent(agent_file)

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

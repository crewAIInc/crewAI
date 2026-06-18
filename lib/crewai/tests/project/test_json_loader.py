"""Tests for crewai.project.json_loader."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from pydantic import BaseModel

from crewai.llms.base_llm import BaseLLM
from crewai.project.json_loader import (
    JSONProjectValidationError,
    _looks_like_windows_absolute_path,
    find_json_project_file,
    load_agent,
    load_agent_from_definition,
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


@pytest.mark.parametrize(
    "path_value",
    [
        r"C:\Users\alice\.ssh\id_rsa",
        "C:/Users/alice/.ssh/id_rsa",
        r"\\server\share\secret.txt",
        "//server/share/secret.txt",
    ],
)
def test_windows_absolute_path_detection(path_value: str):
    assert _looks_like_windows_absolute_path(path_value)


@pytest.mark.parametrize(
    "path_value",
    [
        r"folder\file.txt",
        "folder/file.txt",
        r"\server\share\secret.txt",
    ],
)
def test_windows_absolute_path_detection_ignores_relative_paths(path_value: str):
    assert not _looks_like_windows_absolute_path(path_value)


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

    def test_load_agent_with_llm_config_object(self, tmp_path: Path):
        agent_def = {
            "role": "Coder",
            "goal": "Write code",
            "backstory": "Expert coder.",
            "llm": {
                "model": "llama3",
                "provider": "ollama",
                "temperature": 0.2,
                "base_url": "http://localhost:11434",
            },
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)

        assert isinstance(agent.llm, BaseLLM)
        assert agent.llm.model == "llama3"
        assert agent.llm.provider == "ollama"
        assert agent.llm.temperature == 0.2
        assert agent.llm.base_url == "http://localhost:11434/v1"

    def test_load_agent_with_planning_config_llm_object(self, tmp_path: Path):
        agent_def = {
            "role": "Planner",
            "goal": "Plan work",
            "backstory": "Expert planner.",
            "llm": "ollama/llama3",
            "planning_config": {
                "reasoning_effort": "high",
                "llm": {
                    "model": "deepseek-chat",
                    "provider": "deepseek",
                    "api_key": "test-key",
                },
            },
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)

        assert agent.planning_config is not None
        assert isinstance(agent.planning_config.llm, BaseLLM)
        assert agent.planning_config.llm.model == "deepseek-chat"
        assert agent.planning_config.llm.provider == "deepseek"
        assert agent.planning_config.llm.api_key == "test-key"

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

    def test_load_agent_accepts_static_mcp_tool_filter(self, tmp_path: Path):
        agent_def = {
            "role": "MCP User",
            "goal": "Use MCP tools",
            "backstory": "MCP expert.",
            "mcps": [
                {
                    "command": "python",
                    "args": ["server.py"],
                    "tool_filter": {
                        "type": "static",
                        "allowed_tool_names": ["read_file"],
                        "blocked_tool_names": ["delete_file"],
                    },
                }
            ],
        }
        agent_file = tmp_path / "agent.json"
        agent_file.write_text(json.dumps(agent_def))

        agent = load_agent(agent_file)

        tool_filter = agent.mcps[0].tool_filter
        assert tool_filter({"name": "read_file"})
        assert not tool_filter({"name": "delete_file"})
        assert not tool_filter({"name": "write_file"})

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


class TestLoadAgentFromDefinition:
    def test_loads_inline_agent_and_ignores_kickoff_input(self):
        agent, response_format = load_agent_from_definition(
            {
                "role": "Analyst",
                "goal": "Analyze data",
                "backstory": "Data expert.",
                "settings": {"verbose": True, "max_iter": 3},
                "input": "${state.question}",
            },
            source="agent action",
        )

        assert agent.role == "Analyst"
        assert agent.verbose is True
        assert agent.max_iter == 3
        assert response_format is None

    def test_resolves_response_format(self, tmp_path: Path):
        (tmp_path / "models.py").write_text(
            "from pydantic import BaseModel\n"
            "class AnswerModel(BaseModel):\n"
            "    answer: str\n"
        )
        sys.modules.pop("models", None)

        _, response_format = load_agent_from_definition(
            {
                "role": "Analyst",
                "goal": "Analyze data",
                "backstory": "Data expert.",
                "input": "Summarize this",
                "response_format": {"python": "models.AnswerModel"},
            },
            source="agent action",
            project_root=tmp_path,
        )

        assert response_format is not None
        assert issubclass(response_format, BaseModel)
        assert response_format.__name__ == "AnswerModel"


class TestResolveTools:
    def test_unknown_tool_raises_with_guidance(self):
        from crewai.project.json_loader import JSONProjectError, _resolve_tools

        with pytest.raises(JSONProjectError, match="Unknown tool 'NotARealToolXYZ'"):
            _resolve_tools(["NotARealToolXYZ"])

    def test_missing_custom_tool_raises(self, tmp_path, monkeypatch):
        from crewai.project.json_loader import JSONProjectError, _resolve_tools

        monkeypatch.chdir(tmp_path)
        with pytest.raises(JSONProjectError, match="custom:missing"):
            _resolve_tools(["custom:missing"])

    def test_custom_tool_without_basetool_subclass_raises(self, tmp_path, monkeypatch):
        from crewai.project.json_loader import JSONProjectError, _resolve_tools

        monkeypatch.chdir(tmp_path)
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "empty.py").write_text("x = 1\n")

        with pytest.raises(JSONProjectError, match="No BaseTool subclass"):
            _resolve_tools(["custom:empty"])

    def test_custom_tool_resolves(self, tmp_path, monkeypatch):
        from crewai.project.json_loader import _resolve_tools

        monkeypatch.chdir(tmp_path)
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "echo.py").write_text(
            "from crewai.tools.base_tool import BaseTool\n"
            "\n"
            "class EchoTool(BaseTool):\n"
            "    name: str = 'echo'\n"
            "    description: str = 'echo input'\n"
            "\n"
            "    def _run(self, text: str) -> str:\n"
            "        return text\n"
        )

        tools = _resolve_tools(["custom:echo"])

        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_serialized_tool_dicts_pass_through(self):
        from crewai.project.json_loader import _resolve_tools

        spec = {"tool_type": "some.module.Tool"}
        assert _resolve_tools([spec]) == [spec]


class TestValidationDoesNotExecuteTools:
    def _write_project(self, root, tool_line='"custom:landmine"'):
        agents_dir = root / "agents"
        agents_dir.mkdir()
        (agents_dir / "worker.jsonc").write_text(
            "{\n"
            '  "role": "Worker",\n'
            '  "goal": "Work",\n'
            '  "backstory": "Works hard",\n'
            f'  "tools": [{tool_line}]\n'
            "}\n"
        )
        crew_path = root / "crew.jsonc"
        crew_path.write_text(
            "{\n"
            '  "agents": ["worker"],\n'
            '  "tasks": [\n'
            '    {"name": "t1", "description": "Do work", '
            '"expected_output": "Done", "agent": "worker"}\n'
            "  ]\n"
            "}\n"
        )
        return crew_path

    def test_validate_does_not_execute_custom_tool_code(self, tmp_path):
        from crewai.project.json_loader import validate_crew_project

        sentinel = tmp_path / "executed.txt"
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "landmine.py").write_text(
            f"open({str(sentinel)!r}, 'w').write('boom')\n"
        )
        crew_path = self._write_project(tmp_path)

        project = validate_crew_project(crew_path, tmp_path / "agents")

        assert not sentinel.exists(), "validation must not execute tools/<name>.py"
        assert project.agent_names == ["worker"]

    def test_validate_does_not_import_python_refs(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        from crewai.project.json_loader import validate_crew_project

        sentinel = tmp_path / "python_ref_executed.txt"
        (tmp_path / "callbacks.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('boom')\n"
            "def step_callback(*_args, **_kwargs):\n"
            "    return None\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("callbacks", None)
        crew_path = self._write_project(
            tmp_path,
            tool_line='{"tool_type": "some.module.Tool"}',
        )
        agent_file = tmp_path / "agents" / "worker.jsonc"
        agent_def = json.loads(agent_file.read_text())
        agent_def["step_callback"] = {"python": "callbacks.step_callback"}
        agent_file.write_text(json.dumps(agent_def))

        validate_crew_project(crew_path, tmp_path / "agents")

        assert not sentinel.exists(), "validation must not import Python refs"

    def test_validate_reports_missing_custom_tool_file(self, tmp_path):
        from crewai.project.json_loader import (
            JSONProjectValidationError,
            validate_crew_project,
        )

        crew_path = self._write_project(tmp_path)

        with pytest.raises(JSONProjectValidationError) as exc_info:
            validate_crew_project(crew_path, tmp_path / "agents")

        assert "custom:landmine" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_validate_reports_path_escaping_custom_tool(self, tmp_path):
        from crewai.project.json_loader import (
            JSONProjectValidationError,
            validate_crew_project,
        )

        crew_path = self._write_project(tmp_path, tool_line='"custom:../evil"')

        with pytest.raises(JSONProjectValidationError) as exc_info:
            validate_crew_project(crew_path, tmp_path / "agents")

        assert "Invalid custom tool name" in str(exc_info.value)

    def test_validate_rejects_deep_python_ref_nesting(self, tmp_path):
        from crewai.project.json_loader import validate_crew_project

        crew_path = self._write_project(
            tmp_path,
            tool_line='{"tool_type": "some.module.Tool"}',
        )
        agent_file = tmp_path / "agents" / "worker.jsonc"
        agent_def = json.loads(agent_file.read_text())
        nested: dict[str, object] = {}
        current = nested
        for _ in range(70):
            child: dict[str, object] = {}
            current["nested"] = child
            current = child
        current["ref"] = {"python": "callbacks.step_callback"}
        agent_def["security_config"] = nested
        agent_file.write_text(json.dumps(agent_def))

        with pytest.raises(JSONProjectValidationError, match="maximum depth"):
            validate_crew_project(crew_path, tmp_path / "agents")


class TestCustomToolPathSafety:
    @pytest.mark.parametrize(
        "bad_name",
        ["../evil", "..", "sub/inner", "/etc/passwd", "a-b", "", "name.py"],
    )
    def test_unsafe_names_rejected_at_runtime(self, bad_name, tmp_path, monkeypatch):
        from crewai.project.json_loader import JSONProjectError, _resolve_tools

        monkeypatch.chdir(tmp_path)
        with pytest.raises(JSONProjectError, match="Invalid custom tool name"):
            _resolve_tools([f"custom:{bad_name}"])

    def test_resolves_relative_to_project_root_not_cwd(self, tmp_path, monkeypatch):
        from crewai.project.json_loader import _resolve_tools

        project_root = tmp_path / "project"
        tools_dir = project_root / "tools"
        tools_dir.mkdir(parents=True)
        (tools_dir / "echo.py").write_text(
            "from crewai.tools.base_tool import BaseTool\n"
            "\n"
            "class EchoTool(BaseTool):\n"
            "    name: str = 'echo'\n"
            "    description: str = 'echo input'\n"
            "\n"
            "    def _run(self, text: str) -> str:\n"
            "        return text\n"
        )
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        tools = _resolve_tools(["custom:echo"], project_root=project_root)

        assert len(tools) == 1
        assert tools[0].name == "echo"

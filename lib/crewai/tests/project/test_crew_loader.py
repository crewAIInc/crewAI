"""Tests for crewai.project.crew_loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.project.json_loader import JSONProjectError, JSONProjectValidationError
from crewai.project.crew_loader import load_crew


def _write_agent(agents_dir: Path, name: str, **overrides) -> Path:
    defn = {
        "role": f"{name} role",
        "goal": f"{name} goal",
        "backstory": f"{name} backstory",
    }
    defn.update(overrides)
    f = agents_dir / f"{name}.jsonc"
    f.write_text(json.dumps(defn))
    return f


def _write_crew(project_dir: Path, crew_def: dict) -> Path:
    f = project_dir / "crew.jsonc"
    f.write_text(json.dumps(crew_def))
    return f


class TestLoadCrew:
    def test_minimal_crew(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "researcher")

        crew_def = {
            "name": "test_crew",
            "agents": ["researcher"],
            "tasks": [
                {
                    "name": "research",
                    "description": "Do research",
                    "expected_output": "Research findings",
                    "agent": "researcher",
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, inputs = load_crew(crew_file)
        assert crew.name == "test_crew"
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1
        assert crew.tasks[0].description == "Do research"
        assert inputs == {}

    def test_crew_with_default_inputs(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "researcher")

        crew_def = {
            "name": "test_crew",
            "agents": ["researcher"],
            "tasks": [
                {
                    "name": "research",
                    "description": "Research {topic}",
                    "expected_output": "Findings about {topic}",
                    "agent": "researcher",
                }
            ],
            "inputs": {"topic": "AI"},
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, inputs = load_crew(crew_file)
        assert inputs == {"topic": "AI"}

    def test_crew_with_multiple_agents(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "researcher")
        _write_agent(agents_dir, "writer")

        crew_def = {
            "name": "multi_crew",
            "agents": ["researcher", "writer"],
            "tasks": [
                {
                    "name": "research",
                    "description": "Do research",
                    "expected_output": "Findings",
                    "agent": "researcher",
                },
                {
                    "name": "write",
                    "description": "Write report",
                    "expected_output": "Report",
                    "agent": "writer",
                    "context": ["research"],
                },
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        # Second task should have context referencing first task
        assert crew.tasks[1].context is not None
        assert len(crew.tasks[1].context) == 1

    def test_crew_hierarchical_process(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker")

        crew_def = {
            "name": "hier_crew",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "work",
                    "description": "Do work",
                    "expected_output": "Work done",
                    "agent": "worker",
                }
            ],
            "process": "hierarchical",
            "manager_llm": "openai/gpt-4o",
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        from crewai import Process
        assert crew.process == Process.hierarchical

    def test_crew_accepts_llm_config_objects(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker", llm="ollama/llama3")

        crew_def = {
            "name": "llm_config_crew",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "work",
                    "description": "Do work",
                    "expected_output": "Work done",
                    "agent": "worker",
                }
            ],
            "process": "hierarchical",
            "manager_llm": {
                "model": "llama3",
                "provider": "ollama",
                "base_url": "http://localhost:11434",
            },
            "planning_llm": {
                "model": "deepseek-chat",
                "provider": "deepseek",
                "api_key": "test-key",
            },
            "chat_llm": {
                "model": "openrouter/anthropic/claude-3-opus",
                "api_key": "test-key",
            },
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)

        assert isinstance(crew.manager_llm, BaseLLM)
        assert crew.manager_llm.model == "llama3"
        assert crew.manager_llm.provider == "ollama"
        assert crew.manager_llm.base_url == "http://localhost:11434/v1"
        assert isinstance(crew.planning_llm, BaseLLM)
        assert crew.planning_llm.model == "deepseek-chat"
        assert crew.planning_llm.provider == "deepseek"
        assert isinstance(crew.chat_llm, BaseLLM)
        assert crew.chat_llm.model == "anthropic/claude-3-opus"
        assert crew.chat_llm.provider == "openrouter"

    def test_crew_accepts_public_crew_config_fields(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker")

        crew_def = {
            "name": "config_crew",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "work",
                    "description": "Do work",
                    "expected_output": "Work done",
                    "agent": "worker",
                }
            ],
            "cache": False,
            "max_rpm": 12,
            "planning": True,
            "planning_llm": "openai/gpt-4o-mini",
            "share_crew": False,
            "output_log_file": "crew.log",
            "tracing": False,
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        assert crew.cache is False
        assert crew.max_rpm == 12
        assert crew.planning is True
        assert crew.planning_llm == "openai/gpt-4o-mini"
        assert crew.output_log_file == "crew.log"
        assert crew.tracing is False

    def test_crew_with_output_file(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "writer")

        crew_def = {
            "name": "output_crew",
            "agents": ["writer"],
            "tasks": [
                {
                    "name": "write",
                    "description": "Write something",
                    "expected_output": "Written content",
                    "agent": "writer",
                    "output_file": "output.md",
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        assert crew.tasks[0].output_file == "output.md"

    def test_task_accepts_public_task_config_fields(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "writer")

        schema = {
            "title": "ReportOutput",
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        }
        crew_def = {
            "name": "task_config_crew",
            "agents": ["writer"],
            "tasks": [
                {
                    "name": "write",
                    "description": "Write something",
                    "expected_output": "Written content",
                    "agent": "writer",
                    "output_json": schema,
                    "response_model": schema,
                    "create_directory": False,
                    "human_input": True,
                    "markdown": True,
                    "guardrail": "Return a summary field.",
                    "guardrail_max_retries": 1,
                    "allow_crewai_trigger_context": False,
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        task = crew.tasks[0]
        assert task.output_json is not None
        assert "summary" in task.output_json.model_fields
        assert task.response_model is not None
        assert task.create_directory is False
        assert task.human_input is True
        assert task.markdown is True
        assert task.guardrail == "Return a summary field."
        assert task.allow_crewai_trigger_context is False

    def test_missing_agent_file_raises(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        crew_def = {
            "name": "broken_crew",
            "agents": ["nonexistent"],
            "tasks": [],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_crew(crew_file)

    def test_task_references_unknown_agent_raises(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "researcher")

        crew_def = {
            "name": "bad_ref_crew",
            "agents": ["researcher"],
            "tasks": [
                {
                    "name": "task1",
                    "description": "Do something",
                    "expected_output": "Something",
                    "agent": "unknown_agent",
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        with pytest.raises(JSONProjectError, match="unknown_agent"):
            load_crew(crew_file)

    def test_task_context_order_dependency(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker")

        crew_def = {
            "name": "order_crew",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "task2",
                    "description": "Second task",
                    "expected_output": "Output",
                    "agent": "worker",
                    "context": ["task1"],
                },
                {
                    "name": "task1",
                    "description": "First task",
                    "expected_output": "Output",
                    "agent": "worker",
                },
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        with pytest.raises(JSONProjectError, match="task1"):
            load_crew(crew_file)

    def test_runtime_fields_are_rejected(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker")

        crew_def = {
            "name": "bad_runtime_crew",
            "id": "00000000-0000-4000-8000-000000000000",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "work",
                    "description": "Work",
                    "expected_output": "Done",
                    "agent": "worker",
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        with pytest.raises(JSONProjectValidationError, match="runtime-only"):
            load_crew(crew_file)

    def test_custom_agents_dir(self, tmp_path: Path):
        custom_dir = tmp_path / "my_agents"
        custom_dir.mkdir()
        _write_agent(custom_dir, "analyst")

        crew_def = {
            "name": "custom_dir_crew",
            "agents": ["analyst"],
            "tasks": [
                {
                    "name": "analyze",
                    "description": "Analyze data",
                    "expected_output": "Analysis",
                    "agent": "analyst",
                }
            ],
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file, agents_dir=custom_dir)
        assert len(crew.agents) == 1

    def test_crew_verbose_and_memory_flags(self, tmp_path: Path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        _write_agent(agents_dir, "worker")

        crew_def = {
            "name": "flags_crew",
            "agents": ["worker"],
            "tasks": [
                {
                    "name": "work",
                    "description": "Work",
                    "expected_output": "Done",
                    "agent": "worker",
                }
            ],
            "verbose": True,
            "memory": True,
        }
        crew_file = _write_crew(tmp_path, crew_def)

        crew, _ = load_crew(crew_file)
        assert crew.verbose is True

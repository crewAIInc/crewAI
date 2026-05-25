"""Tests for crewai.project.crew_loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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

        with pytest.raises(KeyError, match="unknown_agent"):
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

        with pytest.raises(KeyError, match="task1"):
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

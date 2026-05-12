"""Tests for the agent TUI and crewai run integration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest


def strip_jsonc_comments(text: str) -> str:
    result = re.sub(r"(?<!:)//.*?$", "", text, flags=re.MULTILINE)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    return result


class TestLoadAgents:
    """Tests for loading agent definitions from agents/ directory."""

    def test_loads_json_file(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        defn = {"name": "test", "role": "Test", "goal": "Test"}
        (agents_dir / "test.json").write_text(json.dumps(defn))

        agents = _load_agents(agents_dir)
        assert len(agents) == 1
        assert agents[0]["name"] == "test"

    def test_loads_jsonc_file(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        jsonc = '{\n  // comment\n  "name": "test",\n  "role": "R",\n  "goal": "G"\n}'
        (agents_dir / "test.jsonc").write_text(jsonc)

        agents = _load_agents(agents_dir)
        assert len(agents) == 1
        assert agents[0]["name"] == "test"

    def test_loads_multiple_agents(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        for name in ("alpha", "beta", "gamma"):
            defn = {"name": name, "role": name.title(), "goal": f"{name} goal"}
            (agents_dir / f"{name}.json").write_text(json.dumps(defn))

        agents = _load_agents(agents_dir)
        assert len(agents) == 3
        names = [a["name"] for a in agents]
        assert sorted(names) == ["alpha", "beta", "gamma"]

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "good.json").write_text('{"name": "good", "role": "R", "goal": "G"}')
        (agents_dir / "bad.json").write_text("this is not json {{{")

        agents = _load_agents(agents_dir)
        assert len(agents) == 1
        assert agents[0]["name"] == "good"

    def test_empty_directory(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_agents

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        agents = _load_agents(agents_dir)
        assert agents == []


class TestLoadConfig:
    """Tests for loading project config.json."""

    def test_loads_config(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_config

        config = {"rooms": {"common": {"agents": ["a", "b"], "engagement": "tagged"}}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        result = _load_config(tmp_path)
        assert result["rooms"]["common"]["engagement"] == "tagged"
        assert result["rooms"]["common"]["agents"] == ["a", "b"]

    def test_missing_config_returns_defaults(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_config

        result = _load_config(tmp_path)
        assert "rooms" in result
        assert "common" in result["rooms"]

    def test_loads_jsonc_config(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import _load_config

        jsonc = '{\n  // comment\n  "rooms": {"common": {"agents": [], "engagement": "organic"}}\n}'
        (tmp_path / "config.json").write_text(jsonc)

        result = _load_config(tmp_path)
        assert result["rooms"]["common"]["engagement"] == "organic"


class TestHasAgentsDir:
    """Tests for _has_agents_dir detection in run_crew."""

    def test_detects_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.run_crew import _has_agents_dir

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "test.json").write_text('{"name": "test"}')

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert _has_agents_dir() is True
        finally:
            os.chdir(old_cwd)

    def test_no_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.run_crew import _has_agents_dir

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert _has_agents_dir() is False
        finally:
            os.chdir(old_cwd)

    def test_empty_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.run_crew import _has_agents_dir

        (tmp_path / "agents").mkdir()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert _has_agents_dir() is False
        finally:
            os.chdir(old_cwd)


class TestAgentTUIConstruction:
    """Tests for AgentTUI class construction."""

    def test_constructs_with_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import AgentTUI

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "test.json").write_text('{"name": "test", "role": "R", "goal": "G"}')

        tui = AgentTUI(agents_dir=agents_dir)
        assert tui._agents_dir == agents_dir

    def test_constructs_with_config(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import AgentTUI

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        config = {"rooms": {"common": {"agents": ["test"], "engagement": "organic"}}}
        tui = AgentTUI(agents_dir=agents_dir, config=config)
        assert tui._config["rooms"]["common"]["engagement"] == "organic"


class TestRunAgentTUI:
    """Tests for run_agent_tui function."""

    def test_exits_if_no_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import run_agent_tui

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(SystemExit):
                run_agent_tui()
        finally:
            os.chdir(old_cwd)

    def test_exits_if_empty_agents_dir(self, tmp_path: Path) -> None:
        from crewai_cli.agent_tui import run_agent_tui

        (tmp_path / "agents").mkdir()

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with pytest.raises(SystemExit):
                run_agent_tui()
        finally:
            os.chdir(old_cwd)

"""Tests for NewAgent CLI commands (create agent, agent reset-history, agent memory)."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from crewai_cli.cli import crewai
from crewai_cli.create_agent import AGENT_TEMPLATE, create_agent


# ── Helpers ─────────────────────────────────────────────────────


def strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments so the output is valid JSON."""
    result = re.sub(r"(?<!:)//.*?$", "", text, flags=re.MULTILINE)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    result = re.sub(r",\s*([}\]])", r"\1", result)
    return result


# ── Helpers ─────────────────────────────────────────────────────

# Standard interactive input for agent creation:
# role, goal, backstory, provider (1=OpenAI), model (1=first), tools (none), api key (skip)
_DEFAULT_PROMPTS_INPUT = "Test Role\nTest Goal\n\n1\n1\n\n\n"


# ── crewai create agent <name> ──────────────────────────────────


class TestCreateAgentCommand:
    """Tests for ``crewai create agent <name>``."""

    def test_creates_jsonc_file(self, tmp_path: Path) -> None:
        """The command should create agents/<name>.jsonc."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                crewai, ["create", "agent", "researcher"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            assert result.exit_code == 0, result.output
            dest = Path("agents/researcher.jsonc")
            assert dest.exists(), f"Expected {dest} to be created"

    def test_file_contains_agent_name(self, tmp_path: Path) -> None:
        """The scaffolded file must contain the agent name."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "writer"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            content = Path("agents/writer.jsonc").read_text()
            assert '"name": "writer"' in content

    def test_prompts_populate_fields(self, tmp_path: Path) -> None:
        """Interactive prompts should populate role, goal, backstory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # role, goal, backstory, provider (1=OpenAI), model (1=first), tools (none), api key (skip)
            result = runner.invoke(
                crewai, ["create", "agent", "analyst"],
                input="Data Analyst\nAnalyze data\nExpert analyst\n1\n1\n\n\n",
            )
            assert result.exit_code == 0, result.output
            raw = Path("agents/analyst.jsonc").read_text()
            clean = strip_jsonc_comments(raw)
            data = json.loads(clean)
            assert data["name"] == "analyst"
            assert data["role"] == "Data Analyst"
            assert data["goal"] == "Analyze data"
            assert data["backstory"] == "Expert analyst"
            assert data["llm"] == "openai/gpt-5.5"

    def test_tools_selection(self, tmp_path: Path) -> None:
        """Selecting tools should populate the tools array."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # role, goal, backstory, provider (1), model (1), tools (1 2 = SerperDevTool + ScrapeWebsiteTool), api key (skip)
            result = runner.invoke(
                crewai, ["create", "agent", "searcher"],
                input="Web Searcher\nSearch things\n\n1\n1\n1 2\n\n",
            )
            assert result.exit_code == 0, result.output
            raw = Path("agents/searcher.jsonc").read_text()
            clean = strip_jsonc_comments(raw)
            data = json.loads(clean)
            assert data["tools"] == ["SerperDevTool", "ScrapeWebsiteTool"]

    def test_jsonc_is_parseable(self, tmp_path: Path) -> None:
        """After stripping comments the JSONC must be valid JSON."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "analyst"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            raw = Path("agents/analyst.jsonc").read_text()
            clean = strip_jsonc_comments(raw)
            data = json.loads(clean)
            assert data["name"] == "analyst"
            assert data["settings"]["memory"] is True
            assert data["settings"]["planning"] is True

    def test_all_expected_fields_present(self, tmp_path: Path) -> None:
        """The scaffolded JSON should contain every documented field."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "myagent"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            raw = Path("agents/myagent.jsonc").read_text()
            data = json.loads(strip_jsonc_comments(raw))
            for key in ("name", "role", "goal", "backstory", "llm", "tools", "mcps", "coworkers", "settings"):
                assert key in data, f"Missing expected field: {key}"

    def test_does_not_overwrite_without_confirm(self, tmp_path: Path) -> None:
        """If the file already exists, declining should leave it untouched."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "dup"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            original = Path("agents/dup.jsonc").read_text()

            # Decline overwrite (input 'n' after the prompts)
            result = runner.invoke(
                crewai, ["create", "agent", "dup"],
                input="n\n",
            )
            assert "cancelled" in result.output.lower()
            assert Path("agents/dup.jsonc").read_text() == original

    def test_creates_agents_directory(self, tmp_path: Path) -> None:
        """The agents/ directory should be created if it does not exist."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            assert not Path("agents").exists()
            runner.invoke(
                crewai, ["create", "agent", "newone"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            assert Path("agents").is_dir()

    def test_success_message(self, tmp_path: Path) -> None:
        """The command should print a success message."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                crewai, ["create", "agent", "bot"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            assert "Agent created:" in result.output


# ── crewai agent reset-history <name> ───────────────────────────


class TestAgentResetHistoryCommand:
    """Tests for ``crewai agent reset-history <name>``."""

    def test_no_history_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "reset-history", "researcher"])
        assert result.exit_code == 0, result.output
        assert "researcher" in result.output
        assert "no conversation history" in result.output.lower()

    def test_deletes_history_file(self, tmp_path: Path) -> None:
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            history_dir = tmp_path / ".crewai" / "conversations"
            history_dir.mkdir(parents=True)
            history_file = history_dir / "test-agent.json"
            history_file.write_text("[]")

            runner = CliRunner()
            result = runner.invoke(crewai, ["agent", "reset-history", "test-agent"])
            assert result.exit_code == 0
            assert "cleared" in result.output.lower()
            assert not history_file.exists()
        finally:
            os.chdir(old_cwd)

    def test_accepts_any_name(self) -> None:
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "reset-history", "my-custom-agent"])
        assert result.exit_code == 0
        assert "my-custom-agent" in result.output


# ── Template unit tests ─────────────────────────────────────────


class TestAgentTemplate:
    """Unit tests for the AGENT_TEMPLATE constant."""

    def _render(self, **kwargs) -> str:
        defaults = {"name": "test", "role": "", "goal": "", "backstory": "", "llm": "openai/gpt-5.5"}
        defaults.update(kwargs)
        return AGENT_TEMPLATE.format(**defaults)

    def test_template_renders_name(self) -> None:
        content = self._render(name="tester")
        assert '"name": "tester"' in content

    def test_template_is_valid_jsonc(self) -> None:
        content = self._render(name="demo")
        clean = strip_jsonc_comments(content)
        data = json.loads(clean)
        assert data["name"] == "demo"
        assert isinstance(data["settings"], dict)

    def test_comments_on_line_above(self) -> None:
        """Comments should be on the line before, not inline with values."""
        content = self._render(name="check")
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comment-only lines and blank lines
            if stripped.startswith("//") or not stripped:
                continue
            # Lines with actual JSON values should NOT have inline comments
            if ":" in stripped and not stripped.startswith("//"):
                # Allow trailing comments only on lines that are JUST comments
                assert "//" not in stripped.split(":")[1] or stripped.strip().startswith("//"), \
                    f"Inline comment found on line {i+1}: {line}"


class TestProjectBootstrap:
    """Tests for project structure creation."""

    def test_creates_project_structure(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "myagent"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            assert Path("agents").is_dir()
            assert Path("tools").is_dir()
            assert Path("config.json").exists()

    def test_config_json_is_valid(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "myagent"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            raw = Path("config.json").read_text()
            clean = strip_jsonc_comments(raw)
            data = json.loads(clean)
            assert "rooms" in data

    def test_agent_added_to_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                crewai, ["create", "agent", "researcher"],
                input=_DEFAULT_PROMPTS_INPUT,
            )
            raw = Path("config.json").read_text()
            clean = strip_jsonc_comments(raw)
            data = json.loads(clean)
            agents = data["rooms"]["common"]["agents"]
            assert "researcher" in agents


# ── GAP-65: Schema validation tests ──────────────────────────


class TestSchemaValidation:
    """Tests for agent definition schema validation (GAP-65)."""

    def test_valid_definition_no_warning(self, tmp_path: Path, caplog) -> None:
        """A valid definition should not produce a validation warning."""
        from crewai.new_agent.definition_parser import parse_agent_definition

        valid = {"role": "Tester", "goal": "Test things", "name": "test"}
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent.definition_parser"):
            result = parse_agent_definition(valid)
        assert result["role"] == "Tester"
        # No validation warning expected (if jsonschema is installed)
        validation_warnings = [
            r for r in caplog.records
            if "validation failed" in r.message.lower()
        ]
        assert len(validation_warnings) == 0

    def test_invalid_definition_warns(self, tmp_path: Path, caplog) -> None:
        """An invalid definition (missing required fields) should log a warning."""
        from crewai.new_agent.definition_parser import parse_agent_definition

        invalid = {"name": "bad-agent"}  # Missing required "role" and "goal"
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent.definition_parser"):
            result = parse_agent_definition(invalid)
        # Should still return the dict (graceful degradation)
        assert result["name"] == "bad-agent"
        # Check for validation warning (only if jsonschema is installed)
        try:
            import jsonschema  # noqa: F401
            validation_warnings = [
                r for r in caplog.records
                if "validation failed" in r.message.lower()
            ]
            assert len(validation_warnings) > 0
        except ImportError:
            pass  # No jsonschema, skip assertion

    def test_additional_properties_warns(self, tmp_path: Path, caplog) -> None:
        """Extra properties should trigger a validation warning."""
        from crewai.new_agent.definition_parser import parse_agent_definition

        defn = {
            "role": "Tester",
            "goal": "Test",
            "unknown_field": "should_warn",
        }
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent.definition_parser"):
            result = parse_agent_definition(defn)
        assert result["role"] == "Tester"
        try:
            import jsonschema  # noqa: F401
            validation_warnings = [
                r for r in caplog.records
                if "validation failed" in r.message.lower()
            ]
            assert len(validation_warnings) > 0
        except ImportError:
            pass

    def test_jsonc_file_validated(self, tmp_path: Path, caplog) -> None:
        """JSONC files should be validated after parsing."""
        from crewai.new_agent.definition_parser import parse_agent_definition

        jsonc_content = """{
          // This is a JSONC file
          "role": "Researcher",
          "goal": "Find answers",
          "name": "researcher"
        }"""
        file_path = tmp_path / "test.jsonc"
        file_path.write_text(jsonc_content, encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="crewai.new_agent.definition_parser"):
            result = parse_agent_definition(file_path)
        assert result["role"] == "Researcher"


# ── GAP-68: Agent memory CLI command tests ─────────────────────


class TestAgentMemoryCommand:
    """Tests for ``crewai agent memory <name>``."""

    def test_agent_not_found(self, tmp_path: Path) -> None:
        """Command should report when agent definition is not found."""
        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(crewai, ["agent", "memory", "nonexistent"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()
        finally:
            os.chdir(old_cwd)

    def test_memory_subcommand_exists(self) -> None:
        """The memory subcommand should be registered."""
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "memory", "--help"])
        assert result.exit_code == 0
        assert "memory" in result.output.lower()

    def test_clear_flag_present(self) -> None:
        """The --clear flag should be accepted."""
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "memory", "--help"])
        assert "--clear" in result.output

    def test_search_flag_present(self) -> None:
        """The --search flag should be accepted."""
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "memory", "--help"])
        assert "--search" in result.output

    def test_limit_flag_present(self) -> None:
        """The --limit flag should be accepted."""
        runner = CliRunner()
        result = runner.invoke(crewai, ["agent", "memory", "--help"])
        assert "--limit" in result.output


# ── GAP-28: Organic mode routing tests ─────────────────────────


class TestOrganicMode:
    """Tests for organic engagement mode (GAP-28)."""

    def test_score_relevance_keyword_match(self) -> None:
        """Agents whose role/goal matches message words should score highest."""
        from crewai_cli.agent_tui import AgentTUI

        app = AgentTUI.__new__(AgentTUI)
        agents = [
            {"name": "researcher", "role": "Web Researcher", "goal": "Find information on the web"},
            {"name": "writer", "role": "Content Writer", "goal": "Write compelling articles"},
        ]
        scored = app._score_relevance("search the web for news", agents)
        assert len(scored) > 0
        names = [a["name"] for a, _ in scored]
        assert names[0] == "researcher"

    def test_score_relevance_no_match_returns_empty(self) -> None:
        """When no keywords match, empty list is returned."""
        from crewai_cli.agent_tui import AgentTUI

        app = AgentTUI.__new__(AgentTUI)
        agents = [
            {"name": "a1", "role": "Alpha", "goal": "Do alpha"},
            {"name": "a2", "role": "Beta", "goal": "Do beta"},
        ]
        scored = app._score_relevance("xyzzy foobar", agents)
        assert len(scored) == 0

    def test_score_relevance_filters_stop_words(self) -> None:
        """Stop words should not cause false matches."""
        from crewai_cli.agent_tui import AgentTUI

        app = AgentTUI.__new__(AgentTUI)
        agents = [
            {"name": "helper", "role": "is a helper", "goal": "the goal"},
        ]
        scored = app._score_relevance("is the", agents)
        assert len(scored) == 0

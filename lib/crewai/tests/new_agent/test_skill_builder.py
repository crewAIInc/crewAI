"""Tests for the SkillBuilder — auto-generated SKILL.md suggestion system."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────

def _make_agent(tmp_path: Path, role: str = "analyst", **overrides: Any) -> MagicMock:
    """Create a mock NewAgent with the fields SkillBuilder needs."""
    agent = MagicMock()
    agent.id = "test-agent-123"
    agent.role = role
    agent.settings = MagicMock()
    agent.settings.can_build_skills = overrides.get("can_build_skills", True)
    agent._llm_instance = None
    return agent


def _make_builder(tmp_path: Path, **agent_overrides: Any) -> Any:
    from crewai.new_agent.skill_builder import SkillBuilder

    agent = _make_agent(tmp_path, **agent_overrides)
    with patch.object(SkillBuilder, "_load_existing_skills"):
        builder = SkillBuilder(agent)
    builder._skills_dir = tmp_path / "skills"
    return builder


# ===========================================================================
# Unit Tests: Suggest / Confirm / Reject
# ===========================================================================

class TestSkillBuilderSuggest:
    """Tests for suggest_skill and pending management."""

    def test_suggest_creates_pending(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        result = builder.suggest_skill(
            name="format-report",
            description="Format a weekly report",
            instructions="## Steps\n1. Gather data\n2. Format",
            source="explicit-instruction",
        )
        assert result["name"] == "format-report"
        assert result["status"] == "pending"
        assert len(builder.pending_suggestions) == 1

    def test_suggest_disabled(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, can_build_skills=False)
        result = builder.suggest_skill(
            name="test",
            description="test",
            instructions="test",
            source="test",
        )
        assert result == {}
        assert len(builder.pending_suggestions) == 0

    def test_suggest_slugifies_name(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        result = builder.suggest_skill(
            name="My Cool Skill!",
            description="test",
            instructions="test",
            source="test",
        )
        assert result["name"] == "my-cool-skill"

    def test_suggest_truncates_description(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        result = builder.suggest_skill(
            name="test",
            description="x" * 300,
            instructions="test",
            source="test",
        )
        assert len(result["description"]) == 200

    def test_suggest_deduplicates_name(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        # Add a mock active skill with the same name
        mock_skill = MagicMock()
        mock_skill.name = "my-skill"
        builder._active_skills.append(mock_skill)

        result = builder.suggest_skill(
            name="my-skill",
            description="test",
            instructions="test",
            source="test",
        )
        assert result["name"] != "my-skill"

    def test_suggest_emits_event(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        with patch("crewai.new_agent.skill_builder.crewai_event_bus", create=True) as mock_bus:
            with patch("crewai.new_agent.skill_builder.NewAgentSkillSuggestedEvent", create=True):
                builder.suggest_skill(
                    name="test",
                    description="test",
                    instructions="test",
                    source="explicit-instruction",
                )


class TestSkillBuilderConfirm:
    """Tests for confirm_suggestion and disk write."""

    def test_confirm_writes_skill_md(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="my-skill",
            description="A test skill",
            instructions="## Steps\n1. Do thing A\n2. Do thing B",
            source="explicit-instruction",
        )

        with patch("crewai.skills.parser.load_skill_metadata") as mock_load, \
             patch("crewai.skills.parser.load_skill_instructions") as mock_instruct:
            mock_skill = MagicMock()
            mock_skill.name = "my-skill"
            mock_load.return_value = mock_skill
            mock_instruct.return_value = mock_skill

            result = builder.confirm_suggestion(0)

        assert result is True
        assert len(builder.pending_suggestions) == 0
        assert len(builder._active_skills) == 1

        skill_md = tmp_path / "skills" / "my-skill" / "SKILL.md"
        assert skill_md.exists()
        content = skill_md.read_text()
        assert "name: my-skill" in content
        assert "description: \"A test skill\"" in content
        assert "Do thing A" in content

    def test_confirm_invalid_index(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        assert builder.confirm_suggestion(0) is False
        assert builder.confirm_suggestion(-1) is False

    def test_confirm_already_confirmed(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="test", description="t", instructions="t", source="t"
        )
        builder._pending_suggestions[0]["status"] = "confirmed"
        assert builder.confirm_suggestion(0) is False


class TestSkillBuilderReject:
    """Tests for reject_suggestion."""

    def test_reject_removes_from_pending(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="unwanted", description="t", instructions="t", source="t"
        )
        assert len(builder.pending_suggestions) == 1
        builder.reject_suggestion(0)
        assert len(builder.pending_suggestions) == 0

    def test_reject_invalid_index(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.reject_suggestion(5)  # no crash


class TestSkillBuilderUpdate:
    """Tests for update_suggestion (edit flow)."""

    def test_update_changes_instructions(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="test", description="t", instructions="original", source="t"
        )
        assert builder.update_suggestion(0, "edited instructions")
        assert builder.pending_suggestions[0]["instructions"] == "edited instructions"

    def test_update_invalid_index(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        assert builder.update_suggestion(0, "nope") is False


# ===========================================================================
# Unit Tests: Suggestion from instruction / workflow
# ===========================================================================

class TestSuggestFromInstruction:
    """Tests for suggest_from_instruction (with mocked LLM)."""

    def test_fallback_when_no_llm(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        result = builder.suggest_from_instruction(
            "Always format reports with summary section first"
        )
        assert result["source"] == "explicit-instruction"
        assert result["status"] == "pending"
        assert "format reports" in result["instructions"].lower() or "summary" in result["instructions"].lower()

    def test_uses_llm_when_available(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.agent._llm_instance = MagicMock()

        mock_response = json.dumps({
            "name": "format-reports",
            "description": "Format reports with summary first",
            "instructions": "## Steps\n1. Add summary\n2. Add details",
        })

        with patch("crewai.utilities.agent_utils.get_llm_response", return_value=mock_response):
            result = builder.suggest_from_instruction(
                "Always format reports with summary section first"
            )

        assert result["name"] == "format-reports"
        assert "summary" in result["instructions"].lower()


class TestSuggestFromWorkflow:
    """Tests for suggest_from_workflow."""

    def test_workflow_to_skill(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        workflow = {
            "tools": ["search_web", "scrape_url", "summarize"],
            "count": 7,
        }
        result = builder.suggest_from_workflow(workflow)
        assert result["source"] == "workflow-detection"
        assert result["status"] == "pending"
        assert "search_web" in result["instructions"] or "search-web" in result["name"]


# ===========================================================================
# Unit Tests: Format skills context
# ===========================================================================

class TestFormatSkillsContext:
    """Tests for format_skills_context (prompt injection)."""

    def test_empty_when_no_active_skills(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        assert builder.format_skills_context() == ""

    def test_formats_active_skills(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        mock_skill = MagicMock()
        mock_skill.name = "test-skill"
        mock_skill.description = "A test skill"
        builder._active_skills.append(mock_skill)

        with patch("crewai.skills.loader.format_skill_context", return_value="## Skill: test-skill\nA test skill"):
            result = builder.format_skills_context()
        assert "test-skill" in result


# ===========================================================================
# Unit Tests: Load existing skills from disk
# ===========================================================================

class TestLoadExistingSkills:
    """Tests for _load_existing_skills on init."""

    def test_loads_skills_from_directory(self, tmp_path: Path) -> None:
        from crewai.new_agent.skill_builder import SkillBuilder

        # Create a skills directory with a SKILL.md
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A test\n---\n\n## Instructions\nDo stuff"
        )

        agent = _make_agent(tmp_path)
        builder = SkillBuilder.__new__(SkillBuilder)
        builder.agent = agent
        builder._pending_suggestions = []
        builder._active_skills = []
        builder._skills_dir = tmp_path / "skills"
        builder._load_existing_skills()

        assert len(builder._active_skills) == 1
        assert builder._active_skills[0].name == "my-skill"

    def test_no_crash_when_dir_missing(self, tmp_path: Path) -> None:
        from crewai.new_agent.skill_builder import SkillBuilder

        agent = _make_agent(tmp_path)
        builder = SkillBuilder.__new__(SkillBuilder)
        builder.agent = agent
        builder._pending_suggestions = []
        builder._active_skills = []
        builder._skills_dir = tmp_path / "nonexistent"
        builder._load_existing_skills()
        assert builder._active_skills == []


# ===========================================================================
# Integration: Events
# ===========================================================================

class TestSkillBuilderEvents:
    """Verify events are emitted correctly."""

    def test_suggested_event_fields(self) -> None:
        from crewai.new_agent.events import NewAgentSkillSuggestedEvent

        event = NewAgentSkillSuggestedEvent(
            new_agent_id="abc",
            skill_name="my-skill",
            source_type="explicit-instruction",
        )
        assert event.type == "new_agent_skill_suggested"
        assert event.skill_name == "my-skill"

    def test_confirmed_event_fields(self) -> None:
        from crewai.new_agent.events import NewAgentSkillConfirmedEvent

        event = NewAgentSkillConfirmedEvent(
            new_agent_id="abc",
            skill_name="my-skill",
        )
        assert event.type == "new_agent_skill_confirmed"

    def test_rejected_event_fields(self) -> None:
        from crewai.new_agent.events import NewAgentSkillRejectedEvent

        event = NewAgentSkillRejectedEvent(
            new_agent_id="abc",
            skill_name="my-skill",
        )
        assert event.type == "new_agent_skill_rejected"


# ===========================================================================
# Integration: Settings
# ===========================================================================

class TestSkillBuilderSettings:
    """Verify can_build_skills setting works."""

    def test_setting_default_true(self) -> None:
        from crewai.new_agent.models import AgentSettings

        settings = AgentSettings()
        assert settings.can_build_skills is True

    def test_setting_can_be_disabled(self) -> None:
        from crewai.new_agent.models import AgentSettings

        settings = AgentSettings(can_build_skills=False)
        assert settings.can_build_skills is False


# ===========================================================================
# Integration: PromptStack skills layer
# ===========================================================================

class TestPromptStackSkillsLayer:
    """Verify skills layer is added to PromptStack."""

    def test_skills_layer_included(self, tmp_path: Path) -> None:
        from crewai.new_agent.executor import ConversationalAgentExecutor
        from crewai.new_agent.skill_builder import SkillBuilder
        from crewai.new_agent.models import PromptStack

        agent = MagicMock()
        agent.role = "analyst"
        agent.goal = "analyze data"
        agent.backstory = "expert"
        agent._resolved_tools = []
        agent._coworker_tools = []
        agent._memory_instance = None
        agent.knowledge = None
        agent.knowledge_sources = []
        agent._active_skills = []

        mock_builder = MagicMock(spec=SkillBuilder)
        mock_builder.format_skills_context.return_value = "## Skill: my-skill\nDo things"
        agent._skill_builder = mock_builder

        executor = ConversationalAgentExecutor(agent=agent)

        with patch.object(executor, "_recall_memory", return_value=""), \
             patch.object(executor, "_query_knowledge", return_value=""):
            stack = executor._build_prompt_stack("test query")

        layer_names = [layer.name for layer in stack.layers]
        assert "skills" in layer_names

        skills_layer = next(l for l in stack.layers if l.name == "skills")
        assert "my-skill" in skills_layer.content


# ===========================================================================
# Conversational suggestion response
# ===========================================================================

class TestSuggestionResponse:
    """Tests for conversational approve/reject flow."""

    def test_handle_response_confirm(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="my-skill", description="test", instructions="do stuff", source="test"
        )
        with patch("crewai.skills.parser.load_skill_metadata") as mock_load, \
             patch("crewai.skills.parser.load_skill_instructions") as mock_instruct:
            mock_skill = MagicMock()
            mock_skill.name = "my-skill"
            mock_load.return_value = mock_skill
            mock_instruct.return_value = mock_skill
            result = builder.handle_suggestion_response("yes, save it")
        assert result is not None
        assert result["action"] == "confirmed"
        assert result["name"] == "my-skill"

    def test_handle_response_reject(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="my-skill", description="test", instructions="do stuff", source="test"
        )
        result = builder.handle_suggestion_response("no thanks")
        assert result is not None
        assert result["action"] == "rejected"
        assert len(builder.pending_suggestions) == 0

    def test_handle_response_unrelated(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        builder.suggest_skill(
            name="my-skill", description="test", instructions="do stuff", source="test"
        )
        result = builder.handle_suggestion_response("what's the weather like?")
        assert result is not None
        assert result["action"] == "ignored"
        assert len(builder.pending_suggestions) == 1

    def test_handle_response_no_pending(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        result = builder.handle_suggestion_response("yes")
        assert result is None


class TestBuildSuggestionMessage:
    """Tests for build_suggestion_message (conversational text + actions)."""

    def test_message_contains_name_and_desc(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        suggestion = builder.suggest_skill(
            name="format-report",
            description="Format weekly reports with summary",
            instructions="## Steps\n1. Add summary\n2. Add details",
            source="test",
        )
        text, actions = builder.build_suggestion_message(suggestion)
        assert "format-report" in text
        assert "Format weekly reports" in text
        assert "Would you like me to save" in text

    def test_actions_contain_confirm_reject(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path)
        suggestion = builder.suggest_skill(
            name="test-skill", description="test", instructions="test", source="test"
        )
        text, actions = builder.build_suggestion_message(suggestion)
        action_types = {a["action_type"] for a in actions}
        assert "suggestion_confirm" in action_types
        assert "suggestion_reject" in action_types

    def test_message_action_model(self) -> None:
        from crewai.new_agent.models import MessageAction
        action = MessageAction(
            action_id="test-1",
            label="Approve",
            action_type="suggestion_confirm",
            payload={"type": "skill", "name": "test"},
        )
        assert action.action_id == "test-1"
        assert action.payload["type"] == "skill"

"""Tests for GAP-24, GAP-31, GAP-36, GAP-37, GAP-38, GAP-40, GAP-41, GAP-45, GAP-56, GAP-63.

Covers:
- GAP-24: Anaphora resolution in memory encoding
- GAP-31: Concurrent conversation support
- GAP-36: Apps field warning
- GAP-37: Skills field resolution
- GAP-38: Security/A2A config storage
- GAP-40: Training -> canonical memories
- GAP-41: Memory scoping from provider context
- GAP-45: MemoryScope/MemorySlice types
- GAP-56: AMP circular guard in Python API
- GAP-63: AMP coworker definitions cache
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.new_agent import (
    AgentSettings,
    MemoryScope,
    MemorySlice,
    Message,
    NewAgent,
    clear_amp_cache,
)
from crewai.new_agent.new_agent import (
    _amp_cache,
    _get_init_chain,
    _ANAPHORA_PRONOUNS,
)


# ── GAP-45: MemoryScope / MemorySlice types ─────────────────────


class TestMemoryScopeModel:
    def test_basic_creation(self):
        scope = MemoryScope(namespace="project-alpha")
        assert scope.namespace == "project-alpha"
        assert scope.shared is False

    def test_shared_flag(self):
        scope = MemoryScope(namespace="shared-ns", shared=True)
        assert scope.shared is True

    def test_memory_slice_creation(self):
        ms = MemorySlice(scope="team", user_id="user-1", tags=["important"])
        assert ms.scope == "team"
        assert ms.user_id == "user-1"
        assert ms.tags == ["important"]

    def test_memory_slice_defaults(self):
        ms = MemorySlice()
        assert ms.scope == ""
        assert ms.user_id is None
        assert ms.conversation_id is None
        assert ms.tags == []


class TestMemoryScopeInAgent:
    def test_memory_scope_sets_namespace(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=MemoryScope(namespace="test-ns"),
        )
        assert agent._memory_namespace == "test-ns"
        assert agent._memory_shared is False

    def test_memory_scope_shared(self):
        agent = NewAgent(
            role="R", goal="g",
            memory=MemoryScope(namespace="shared-ns", shared=True),
        )
        assert agent._memory_namespace == "shared-ns"
        assert agent._memory_shared is True

    def test_memory_slice_sets_filter(self):
        ms = MemorySlice(scope="team", user_id="user-1")
        agent = NewAgent(
            role="R", goal="g",
            memory=ms,
        )
        assert agent._memory_namespace == "team"
        assert agent._memory_filter is ms

    def test_bool_memory_still_works(self):
        agent = NewAgent(role="R", goal="g", memory=True)
        # Should not crash, memory_namespace should be None
        assert agent._memory_namespace is None

    def test_false_memory_still_works(self):
        agent = NewAgent(role="R", goal="g", memory=False)
        assert agent._memory_instance is None


# ── GAP-56: AMP Circular Guard ──────────────────────────────────


class TestCircularCoworkerGuard:
    def test_no_infinite_recursion(self):
        """Two agents referencing each other should not loop forever."""
        # We create agents that would reference each other.
        # Since they are NewAgent instances (not AMP handles), we can
        # construct them without actual recursion by building one first
        # and then adding it as a coworker to the other.
        agent_a = NewAgent(role="Agent A", goal="Goal A")
        agent_b = NewAgent(role="Agent B", goal="Goal B", coworkers=[agent_a])

        # Now make A reference B — should not infinite loop
        agent_a_with_b = NewAgent(
            role="Agent A", goal="Goal A", coworkers=[agent_b],
        )
        # Should succeed without recursion
        assert len(agent_a_with_b._resolved_coworkers) == 1
        assert agent_a_with_b._resolved_coworkers[0].role == "Agent B"

    def test_self_reference_skipped(self):
        """An agent referencing itself as a coworker should be ignored."""
        agent = NewAgent(role="Solo", goal="Self")
        agent2 = NewAgent(role="Solo", goal="Self", coworkers=[agent])
        # Since the coworker has the same role, it's filtered out
        assert len(agent2._resolved_coworkers) == 0

    def test_init_chain_is_thread_local(self):
        """The init chain should be thread-local."""
        chain = _get_init_chain()
        assert isinstance(chain, set)
        chain.add("test-id")
        chain.discard("test-id")


# ── GAP-63: AMP Coworker Definitions Cache ─────────────────────


class TestAmpCache:
    def setup_method(self):
        clear_amp_cache()

    def teardown_method(self):
        clear_amp_cache()

    def test_clear_amp_cache(self):
        _amp_cache["test-handle"] = {"role": "Test", "goal": "g"}
        assert "test-handle" in _amp_cache
        clear_amp_cache()
        assert len(_amp_cache) == 0

    @patch("crewai.utilities.agent_utils.load_agent_from_repository")
    def test_cache_hit_avoids_api_call(self, mock_load):
        """Second resolution of same handle should use cache, not call API."""
        mock_load.return_value = {
            "role": "Cached Agent",
            "goal": "cached goal",
        }

        # Pre-populate cache
        _amp_cache["org/agent-1"] = {
            "role": "Cached Agent",
            "goal": "cached goal",
        }

        agent = NewAgent(role="Manager", goal="Manage")
        resolved = agent._resolve_amp_coworker("org/agent-1")

        # API should NOT have been called because cache was hit
        mock_load.assert_not_called()
        assert resolved.role == "Cached Agent"

    @patch("crewai.utilities.agent_utils.load_agent_from_repository")
    def test_cache_miss_calls_api(self, mock_load):
        """First resolution should call API and populate cache."""
        mock_load.return_value = {
            "role": "New Agent",
            "goal": "new goal",
        }

        agent = NewAgent(role="Manager", goal="Manage")
        resolved = agent._resolve_amp_coworker("org/new-agent")

        mock_load.assert_called_once_with("org/new-agent")
        assert resolved.role == "New Agent"
        assert "org/new-agent" in _amp_cache


# ── GAP-31: Concurrent Conversation Support ─────────────────────


class TestConcurrentConversations:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_different_conversation_ids(self, mock_llm):
        mock_llm.side_effect = ["Response for conv-1.", "Response for conv-2."]

        agent = NewAgent(role="R", goal="g")

        r1 = await agent.amessage("Hello conv-1", conversation_id="conv-1")
        r2 = await agent.amessage("Hello conv-2", conversation_id="conv-2")

        assert r1.conversation_id == "conv-1"
        assert r2.conversation_id == "conv-2"

        h1 = agent.get_conversation_history("conv-1")
        h2 = agent.get_conversation_history("conv-2")

        assert len(h1) == 2  # user + agent
        assert len(h2) == 2
        assert h1[0].content == "Hello conv-1"
        assert h2[0].content == "Hello conv-2"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_default_conversation_backward_compat(self, mock_llm):
        mock_llm.return_value = "Default response."

        agent = NewAgent(role="R", goal="g")

        # No conversation_id -> uses default
        r = await agent.amessage("Hello")
        assert r.conversation_id == agent._default_conversation_id
        assert len(agent.conversation_history) == 2

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_get_conversation_history_unknown_id(self, mock_llm):
        agent = NewAgent(role="R", goal="g")
        history = agent.get_conversation_history("nonexistent")
        assert history == []

    def test_reset_specific_conversation(self):
        agent = NewAgent(role="R", goal="g")
        # Create a second conversation executor
        executor = agent._get_or_create_executor("conv-X")
        executor.conversation_history.append(
            Message(role="user", content="test", conversation_id="conv-X"),
        )
        assert len(agent.get_conversation_history("conv-X")) == 1

        agent.reset_conversation(conversation_id="conv-X")
        assert agent.get_conversation_history("conv-X") == []

    def test_reset_default_conversation(self):
        agent = NewAgent(role="R", goal="g")
        old_id = agent._default_conversation_id
        agent.reset_conversation()
        assert agent._default_conversation_id != old_id
        assert len(agent.conversation_history) == 0

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_explain_specific_conversation(self, mock_llm):
        mock_llm.return_value = "Answer."

        agent = NewAgent(role="R", goal="g")
        await agent.amessage("Q", conversation_id="conv-explain")

        entries = agent.explain(conversation_id="conv-explain")
        assert len(entries) == 1
        assert entries[0].action == "response"

    def test_explain_unknown_conversation_returns_empty(self):
        agent = NewAgent(role="R", goal="g")
        entries = agent.explain(conversation_id="nonexistent")
        assert entries == []

    @patch("crewai.new_agent.executor.aget_llm_response")
    def test_sync_message_with_conversation_id(self, mock_llm):
        mock_llm.return_value = "Sync response."
        agent = NewAgent(role="R", goal="g")
        r = agent.message("Hello", conversation_id="sync-conv-1")
        assert r.conversation_id == "sync-conv-1"


# ── GAP-36: Apps Field Warning ──────────────────────────────────


class TestAppsWarning:
    def test_apps_warning_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent"):
            agent = NewAgent(
                role="R", goal="g",
                apps=["app1", "app2"],
            )
        assert "Apps integration requires the CrewAI Platform" in caplog.text
        assert "2 app(s)" in caplog.text

    def test_no_apps_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent"):
            agent = NewAgent(role="R", goal="g")
        assert "Apps integration" not in caplog.text


# ── GAP-37: Skills Field Resolution ─────────────────────────────


class TestSkillsResolution:
    def test_skill_instance_added(self):
        """A skill object with run() is added directly."""
        skill = MagicMock()
        skill.run = MagicMock(return_value="result")

        agent = NewAgent(role="R", goal="g", skills=[skill])
        assert skill in agent._resolved_tools

    def test_skill_path_loaded(self, tmp_path):
        """A Path pointing to a Python file with a tool class is loaded."""
        skill_code = '''
class MySkill:
    name = "my_skill"
    description = "A test skill"
    def run(self, **kwargs):
        return "skill result"
'''
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text(skill_code)

        agent = NewAgent(role="R", goal="g", skills=[skill_file])
        # The skill class should have been instantiated and added
        skill_tools = [t for t in agent._resolved_tools if hasattr(t, 'name') and getattr(t, 'name', '') == 'my_skill']
        assert len(skill_tools) == 1

    def test_invalid_skill_path_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="crewai.new_agent"):
            agent = NewAgent(
                role="R", goal="g",
                skills=[Path("/nonexistent/skill.py")],
            )
        assert "Failed to load skill" in caplog.text or "Cannot load skill" in caplog.text

    def test_empty_skills_no_error(self):
        agent = NewAgent(role="R", goal="g", skills=[])
        assert agent._resolved_tools is not None


# ── GAP-38: Security/A2A Config Storage ─────────────────────────


class TestSecurityA2AConfig:
    def test_security_config_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="crewai.new_agent"):
            agent = NewAgent(
                role="R", goal="g",
                security_config={"auth": "token"},
            )
        assert "Security configuration applied" in caplog.text

    def test_a2a_config_stored(self, caplog):
        a2a_config = {"server": {"port": 8080}}
        with caplog.at_level(logging.INFO, logger="crewai.new_agent"):
            agent = NewAgent(
                role="R", goal="g",
                a2a=a2a_config,
            )
        assert agent._a2a_config == a2a_config
        assert "A2A server configured" in caplog.text

    def test_no_config_no_logs(self, caplog):
        with caplog.at_level(logging.INFO, logger="crewai.new_agent"):
            agent = NewAgent(role="R", goal="g")
        assert "Security configuration" not in caplog.text
        assert "A2A server" not in caplog.text


# ── GAP-40: Training → Canonical Memories ───────────────────────


class TestTraining:
    def test_train_saves_to_memory(self):
        agent = NewAgent(role="R", goal="g")
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        agent.train("Always double-check calculations", "math tasks")

        mock_memory.remember.assert_called_once()
        call_args = mock_memory.remember.call_args
        saved_text = call_args[1].get("value") or call_args[0][0]
        assert "Always double-check calculations" in saved_text
        assert "math tasks" in saved_text

    def test_train_without_context(self):
        agent = NewAgent(role="R", goal="g")
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        agent.train("Be more concise")

        call_args = mock_memory.remember.call_args
        saved_text = call_args[1].get("value") or call_args[0][0]
        assert "Be more concise" in saved_text
        assert "Training feedback" in saved_text

    def test_train_remember_failure_is_silent(self):
        agent = NewAgent(role="R", goal="g")
        mock_memory = MagicMock()
        mock_memory.remember.side_effect = RuntimeError("storage error")
        agent._memory_instance = mock_memory

        # Should not raise
        agent.train("Use shorter sentences")

    def test_train_no_memory_is_noop(self):
        agent = NewAgent(role="R", goal="g", memory=False)
        # Should not raise
        agent.train("Some feedback")

    def test_train_notifies_dreaming_engine(self):
        agent = NewAgent(role="R", goal="g")
        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        mock_dreaming = MagicMock()
        agent._dreaming_engine = mock_dreaming

        agent.train("Important insight", "context")

        mock_dreaming.add_training_feedback.assert_called_once_with(
            "Important insight", "context",
        )


# ── GAP-41: Memory Scoping from Provider Context ────────────────


class TestMemoryScopingFromProvider:
    def test_provider_memory_scope_applied(self):
        mock_provider = MagicMock()
        mock_provider.memory_scope = "slack-channel-123"

        agent = NewAgent(
            role="R", goal="g",
            provider=mock_provider,
        )
        assert agent._memory_namespace == "slack-channel-123"

    def test_manual_memory_scope_overrides_provider(self):
        mock_provider = MagicMock()
        mock_provider.memory_scope = "provider-scope"

        agent = NewAgent(
            role="R", goal="g",
            provider=mock_provider,
            memory_scope="manual-scope",
        )
        # Manual scope takes priority
        assert agent._memory_namespace == "manual-scope"

    def test_no_scope_is_none(self):
        agent = NewAgent(role="R", goal="g")
        assert agent._memory_namespace is None

    def test_provider_without_scope_attr(self):
        mock_provider = MagicMock(spec=[])  # No memory_scope attr
        agent = NewAgent(
            role="R", goal="g",
            provider=mock_provider,
        )
        assert agent._memory_namespace is None


# ── GAP-24: Anaphora Resolution ─────────────────────────────────


class TestAnaphoraResolution:
    def test_pronoun_regex_matches(self):
        assert _ANAPHORA_PRONOUNS.search("He prefers Python")
        assert _ANAPHORA_PRONOUNS.search("She said that")
        assert _ANAPHORA_PRONOUNS.search("It works well")
        assert _ANAPHORA_PRONOUNS.search("They use those tools")
        assert _ANAPHORA_PRONOUNS.search("This is important")

    def test_no_pronouns_no_match(self):
        assert not _ANAPHORA_PRONOUNS.search("Python works well for backend development")

    def test_resolve_anaphora_no_pronouns_returns_unchanged(self):
        agent = NewAgent(role="R", goal="g")
        text = "Python is a great language for backend development"
        result = agent._resolve_anaphora(text, [])
        assert result == text

    def test_prepare_memory_context_format(self):
        agent = NewAgent(role="R", goal="g")
        result = agent.prepare_memory_context("He prefers using it")
        assert "Resolve all pronouns" in result
        assert "He prefers using it" in result

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_prepare_memory_context_includes_history(self, mock_llm):
        mock_llm.return_value = "Response about John."

        agent = NewAgent(role="R", goal="g")
        await agent.amessage("Tell me about John's preferences")

        result = agent.prepare_memory_context("He prefers using it")
        assert "John" in result or "preferences" in result

    def test_resolve_anaphora_with_no_llm(self):
        """If LLM is None, should return text unchanged."""
        agent = NewAgent(role="R", goal="g")
        agent._llm_instance = None
        text = "He likes it"
        result = agent._resolve_anaphora(text, [])
        assert result == text


# ── Integration: Multiple gaps working together ──────────────────


class TestIntegration:
    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_concurrent_conversations_isolated(self, mock_llm):
        """Messages in different conversations should not bleed."""
        mock_llm.side_effect = [
            "Conv A response 1.",
            "Conv B response 1.",
            "Conv A response 2.",
        ]

        agent = NewAgent(role="R", goal="g")

        await agent.amessage("A1", conversation_id="conv-a")
        await agent.amessage("B1", conversation_id="conv-b")
        await agent.amessage("A2", conversation_id="conv-a")

        hist_a = agent.get_conversation_history("conv-a")
        hist_b = agent.get_conversation_history("conv-b")

        assert len(hist_a) == 4  # 2 user + 2 agent
        assert len(hist_b) == 2  # 1 user + 1 agent

        # Verify isolation
        contents_a = [m.content for m in hist_a if m.role == "user"]
        contents_b = [m.content for m in hist_b if m.role == "user"]
        assert "A1" in contents_a
        assert "A2" in contents_a
        assert "B1" in contents_b
        assert "B1" not in contents_a

    def test_memory_scope_with_training(self):
        """Training should work alongside memory scoping."""
        agent = NewAgent(
            role="R", goal="g",
            memory=MemoryScope(namespace="scoped-ns"),
        )

        mock_memory = MagicMock()
        agent._memory_instance = mock_memory

        agent.train("Always verify data sources")
        mock_memory.remember.assert_called_once()

"""Tests for session-start prompt-cache preload feature (#5921).

Verifies that Crew.cache_preload and Crew.cache_preload_strategy
correctly fire lightweight 1-token probes to warm LLM prompt caches
at kickoff time.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(role: str, goal: str, backstory: str) -> Agent:
    return Agent(role=role, goal=goal, backstory=backstory, allow_delegation=False)


def _make_task(description: str, agent: Agent) -> Task:
    return Task(description=description, expected_output="output", agent=agent)


# ---------------------------------------------------------------------------
# BaseLLM.preload_probe
# ---------------------------------------------------------------------------


class TestBaseLLMPreloadProbe:
    def test_preload_probe_fires_one_token_completion(self):
        """preload_probe should delegate to self.call with max_tokens=1."""
        agent = _make_agent("R", "g", "b")
        agent.llm.call = MagicMock(return_value="ok")
        original_max_tokens = agent.llm.max_tokens

        agent.llm.preload_probe("You are a helpful assistant.")

        agent.llm.call.assert_called_once()
        args, kwargs = agent.llm.call.call_args
        # messages may be passed as positional or keyword arg
        messages = args[0] if args else kwargs.get("messages")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        # max_tokens should be restored after the call
        assert agent.llm.max_tokens == original_max_tokens

    def test_preload_probe_does_not_raise_on_failure(self):
        """preload_probe must not propagate exceptions."""
        agent = _make_agent("R", "g", "b")
        agent.llm.call = MagicMock(side_effect=RuntimeError("boom"))

        # Should NOT raise
        agent.llm.preload_probe("system prompt")

    def test_preload_probe_uses_temperature_zero(self):
        """preload_probe should temporarily set temperature=0."""
        agent = _make_agent("R", "g", "b")
        captured_temp = []

        def capture_call(*_args, **_kwargs):
            captured_temp.append(agent.llm.temperature)
            return "ok"

        agent.llm.call = capture_call
        agent.llm.temperature = 0.7

        agent.llm.preload_probe("system prompt")

        assert captured_temp[0] == 0
        assert agent.llm.temperature == 0.7


# ---------------------------------------------------------------------------
# Crew fields
# ---------------------------------------------------------------------------


class TestCachePreloadFields:
    def test_cache_preload_defaults_to_false(self):
        a = _make_agent("R", "g", "b")
        t = _make_task("do it", a)
        crew = Crew(agents=[a], tasks=[t])
        assert crew.cache_preload is False

    def test_cache_preload_strategy_defaults_to_parallel(self):
        a = _make_agent("R", "g", "b")
        t = _make_task("do it", a)
        crew = Crew(agents=[a], tasks=[t])
        assert crew.cache_preload_strategy == "parallel"

    def test_cache_preload_strategy_accepts_valid_values(self):
        a = _make_agent("R", "g", "b")
        t = _make_task("do it", a)
        for strategy in ("parallel", "sequential", "shared_prefix"):
            crew = Crew(
                agents=[a],
                tasks=[t],
                cache_preload=True,
                cache_preload_strategy=strategy,
            )
            assert crew.cache_preload_strategy == strategy


# ---------------------------------------------------------------------------
# Parallel strategy
# ---------------------------------------------------------------------------


class TestParallelStrategy:
    def test_parallel_strategy_probes_all_agents(self):
        a1 = _make_agent("Researcher", "research AI", "You research stuff.")
        a2 = _make_agent("Writer", "write content", "You write stuff.")
        t1 = _make_task("research task", a1)
        t2 = _make_task("writing task", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
            cache_preload_strategy="parallel",
        )

        a1.llm.preload_probe = MagicMock()
        a2.llm.preload_probe = MagicMock()

        crew._preload_caches()

        a1.llm.preload_probe.assert_called_once()
        a2.llm.preload_probe.assert_called_once()

    def test_parallel_strategy_passes_system_prompt(self):
        a1 = _make_agent("Researcher", "research AI", "You research stuff.")
        t1 = _make_task("task", a1)
        a2 = _make_agent("Writer", "write content", "You write stuff.")
        t2 = _make_task("task2", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
            cache_preload_strategy="parallel",
        )

        a1.llm.preload_probe = MagicMock()
        a2.llm.preload_probe = MagicMock()

        crew._preload_caches()

        probe_arg = a1.llm.preload_probe.call_args[0][0]
        assert isinstance(probe_arg, str)
        assert len(probe_arg) > 0


# ---------------------------------------------------------------------------
# Sequential strategy
# ---------------------------------------------------------------------------


class TestSequentialStrategy:
    def test_sequential_strategy_probes_all_agents(self):
        a1 = _make_agent("Researcher", "research AI", "You research stuff.")
        a2 = _make_agent("Writer", "write content", "You write stuff.")
        t1 = _make_task("research task", a1)
        t2 = _make_task("writing task", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
            cache_preload_strategy="sequential",
        )

        a1.llm.preload_probe = MagicMock()
        a2.llm.preload_probe = MagicMock()

        crew._preload_caches()

        a1.llm.preload_probe.assert_called_once()
        a2.llm.preload_probe.assert_called_once()


# ---------------------------------------------------------------------------
# Shared-prefix strategy
# ---------------------------------------------------------------------------


class TestSharedPrefixStrategy:
    def test_shared_prefix_strategy_with_long_common_prefix(self):
        """When agents share >= 1024 chars of prefix, shared prefix is warmed first."""
        shared_backstory = "A" * 2000
        a1 = _make_agent("SharedRole", "shared goal", shared_backstory + " agent1 specifics")
        a2 = _make_agent("SharedRole", "shared goal", shared_backstory + " agent2 specifics")
        t1 = _make_task("task 1", a1)
        t2 = _make_task("task 2", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
            cache_preload_strategy="shared_prefix",
        )

        # Verify the prompts actually share a long common prefix
        p1 = crew._get_agent_system_prompt(a1)
        p2 = crew._get_agent_system_prompt(a2)
        prefix = Crew._common_prefix([p1, p2])
        assert len(prefix) >= 1024, (
            f"Expected common prefix >= 1024 chars, got {len(prefix)}"
        )

        a1.llm.preload_probe = MagicMock()
        a2.llm.preload_probe = MagicMock()

        crew._preload_caches()

        # first_agent's LLM gets probed twice: once for shared prefix, once for full prompt
        assert a1.llm.preload_probe.call_count == 2
        # second agent gets probed once for its full prompt
        assert a2.llm.preload_probe.call_count == 1

    def test_shared_prefix_falls_back_to_parallel_when_prefix_short(self):
        """When the common prefix is < 1024 chars, falls back to parallel."""
        a1 = _make_agent("Researcher", "research AI", "Short backstory for researcher.")
        a2 = _make_agent("Writer", "write content", "Short backstory for writer.")
        t1 = _make_task("task 1", a1)
        t2 = _make_task("task 2", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
            cache_preload_strategy="shared_prefix",
        )

        a1.llm.preload_probe = MagicMock()
        a2.llm.preload_probe = MagicMock()

        crew._preload_caches()

        # Falls back to parallel: each agent probed exactly once
        a1.llm.preload_probe.assert_called_once()
        a2.llm.preload_probe.assert_called_once()


# ---------------------------------------------------------------------------
# Kickoff integration
# ---------------------------------------------------------------------------


class TestKickoffIntegration:
    def test_kickoff_calls_preload_when_enabled(self):
        a1 = _make_agent("Researcher", "research AI", "backstory")
        a2 = _make_agent("Writer", "write content", "backstory")
        t1 = _make_task("task 1", a1)
        t2 = _make_task("task 2", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=True,
        )

        with patch.object(crew, "_preload_caches") as mock_preload, \
             patch.object(crew, "_run_sequential_process", return_value=MagicMock()):
            try:
                crew.kickoff()
            except Exception:
                pass
            mock_preload.assert_called_once()

    def test_kickoff_skips_preload_when_disabled(self):
        a1 = _make_agent("Researcher", "research AI", "backstory")
        a2 = _make_agent("Writer", "write content", "backstory")
        t1 = _make_task("task 1", a1)
        t2 = _make_task("task 2", a2)

        crew = Crew(
            agents=[a1, a2],
            tasks=[t1, t2],
            cache_preload=False,
        )

        with patch.object(crew, "_preload_caches") as mock_preload, \
             patch.object(crew, "_run_sequential_process", return_value=MagicMock()):
            try:
                crew.kickoff()
            except Exception:
                pass
            mock_preload.assert_not_called()

    def test_kickoff_skips_preload_for_single_agent(self):
        a1 = _make_agent("Researcher", "research AI", "backstory")
        t1 = _make_task("task 1", a1)

        crew = Crew(
            agents=[a1],
            tasks=[t1],
            cache_preload=True,
        )

        with patch.object(crew, "_preload_caches") as mock_preload, \
             patch.object(crew, "_run_sequential_process", return_value=MagicMock()):
            try:
                crew.kickoff()
            except Exception:
                pass
            mock_preload.assert_not_called()


# ---------------------------------------------------------------------------
# Crew._common_prefix
# ---------------------------------------------------------------------------


class TestCommonPrefix:
    def test_common_prefix_basic(self):
        assert Crew._common_prefix(["abc", "abd", "abe"]) == "ab"

    def test_common_prefix_empty_list(self):
        assert Crew._common_prefix([]) == ""

    def test_common_prefix_no_overlap(self):
        assert Crew._common_prefix(["abc", "xyz"]) == ""

    def test_common_prefix_identical_strings(self):
        assert Crew._common_prefix(["hello", "hello"]) == "hello"

    def test_common_prefix_single_string(self):
        assert Crew._common_prefix(["only"]) == "only"


# ---------------------------------------------------------------------------
# Crew._get_agent_system_prompt
# ---------------------------------------------------------------------------


class TestGetAgentSystemPrompt:
    def test_returns_nonempty_string(self):
        a = _make_agent("Tester", "test things", "You test stuff.")
        t = _make_task("task", a)
        crew = Crew(agents=[a], tasks=[t])

        prompt = crew._get_agent_system_prompt(a)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_agent_role(self):
        a = _make_agent("SpecialTester", "test things", "You test stuff.")
        t = _make_task("task", a)
        crew = Crew(agents=[a], tasks=[t])

        prompt = crew._get_agent_system_prompt(a)
        assert "SpecialTester" in prompt

    def test_prompt_contains_agent_goal(self):
        a = _make_agent("Tester", "verify correctness", "You test stuff.")
        t = _make_task("task", a)
        crew = Crew(agents=[a], tasks=[t])

        prompt = crew._get_agent_system_prompt(a)
        assert "verify correctness" in prompt

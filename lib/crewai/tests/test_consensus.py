"""Tests for the consensus module and ``Process.consensual`` integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any
from unittest.mock import patch

from crewai.agent import Agent
from crewai.consensus import (
    MAX_PROMPT_TEXT_CHARS,
    ConsensusEngine,
    MajorityVoteConsensus,
    _validate_ballots,
    build_handler_ranking_prompt,
    discover_engines,
    parse_role_ranking,
)
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.process import Process
from crewai.task import Task
from crewai.types.usage_metrics import UsageMetrics
import pytest


def _agent(role: str) -> Agent:
    return Agent(
        role=role,
        goal=f"goal of {role}",
        backstory=f"backstory of {role}",
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# MajorityVoteConsensus
# ---------------------------------------------------------------------------


class TestMajorityVoteConsensus:
    def test_single_voter_winner(self) -> None:
        engine = MajorityVoteConsensus()
        winner = engine.aggregate(["a", "b"], {"v1": ["b", "a"]})
        assert winner == "b"

    def test_majority_wins(self) -> None:
        engine = MajorityVoteConsensus()
        rankings = {
            "v1": ["a", "b", "c"],
            "v2": ["a", "c", "b"],
            "v3": ["b", "a", "c"],
        }
        assert engine.aggregate(["a", "b", "c"], rankings) == "a"

    def test_tie_broken_by_candidate_order(self) -> None:
        engine = MajorityVoteConsensus()
        rankings = {"v1": ["b", "a"], "v2": ["a", "b"]}
        # Both "a" and "b" have one top-1 vote — the first candidate wins.
        assert engine.aggregate(["a", "b"], rankings) == "a"
        # And reversing the candidate order flips the winner.
        assert engine.aggregate(["b", "a"], rankings) == "b"

    def test_empty_rankings_raises(self) -> None:
        engine = MajorityVoteConsensus()
        with pytest.raises(ValueError, match="at least one ranking"):
            engine.aggregate(["a"], {})

    def test_empty_ballot_raises(self) -> None:
        engine = MajorityVoteConsensus()
        with pytest.raises(ValueError, match="empty ranking"):
            engine.aggregate(["a", "b"], {"v1": []})

    def test_unknown_candidate_raises(self) -> None:
        engine = MajorityVoteConsensus()
        with pytest.raises(ValueError, match="unknown candidates"):
            engine.aggregate(["a", "b"], {"v1": ["a", "z"]})

    def test_runtime_checkable_protocol(self) -> None:
        # The Protocol must be marked @runtime_checkable so the Crew
        # validator's isinstance() check works against duck-typed engines.
        assert isinstance(MajorityVoteConsensus(), ConsensusEngine)

        class _NotAnEngine:
            pass

        assert not isinstance(_NotAnEngine(), ConsensusEngine)


# ---------------------------------------------------------------------------
# _validate_ballots
# ---------------------------------------------------------------------------


class TestValidateBallots:
    def test_accepts_complete_ballot(self) -> None:
        _validate_ballots(["a", "b"], {"v1": ["a", "b"]})

    def test_rejects_empty_rankings(self) -> None:
        with pytest.raises(ValueError, match="at least one ranking"):
            _validate_ballots(["a"], {})

    def test_rejects_empty_per_voter_ballot(self) -> None:
        with pytest.raises(ValueError, match="empty ranking"):
            _validate_ballots(["a"], {"v1": []})

    def test_rejects_unknown_candidate(self) -> None:
        with pytest.raises(ValueError, match="unknown candidates"):
            _validate_ballots(["a"], {"v1": ["a", "z"]})


# ---------------------------------------------------------------------------
# parse_role_ranking
# ---------------------------------------------------------------------------


class TestParseRoleRanking:
    def test_strict_json_array(self) -> None:
        assert parse_role_ranking('["b", "a"]', ["a", "b"]) == ["b", "a"]

    def test_json_array_inside_text(self) -> None:
        response = 'Here you go: ["c", "a", "b"] — done.'
        assert parse_role_ranking(response, ["a", "b", "c"]) == ["c", "a", "b"]

    def test_first_appearance_fallback(self) -> None:
        response = "I think Bob is best, then Alice, then Carol."
        assert parse_role_ranking(response, ["Alice", "Bob", "Carol"]) == [
            "Bob",
            "Alice",
            "Carol",
        ]

    def test_partial_json_falls_back_to_text(self) -> None:
        # JSON array doesn't cover all options — parser must fall through
        # to the first-appearance scan and still return a complete ranking.
        response = '["a"] — Alice goes first, then Bob.'
        assert parse_role_ranking(response, ["Alice", "Bob"]) == ["Alice", "Bob"]

    def test_unparseable_response_raises(self) -> None:
        with pytest.raises(ValueError, match="could not extract"):
            parse_role_ranking("nothing useful here", ["Alice", "Bob"])

    def test_partial_text_match_raises(self) -> None:
        # Only "Alice" appears — incomplete ranking, should raise.
        with pytest.raises(ValueError, match="could not extract"):
            parse_role_ranking("Alice is great", ["Alice", "Bob"])


# ---------------------------------------------------------------------------
# build_handler_ranking_prompt
# ---------------------------------------------------------------------------


class TestBuildHandlerRankingPrompt:
    def test_includes_task_and_roles(self) -> None:
        prompt = build_handler_ranking_prompt("write a poem", ["poet", "editor"])
        assert "<task>write a poem</task>" in prompt
        assert json.loads(prompt.rsplit("Roles: ", 1)[1]) == ["poet", "editor"]

    def test_marks_task_as_untrusted(self) -> None:
        prompt = build_handler_ranking_prompt("hi", ["a"])
        assert "UNTRUSTED" in prompt

    def test_caps_long_task_descriptions(self) -> None:
        long_desc = "x" * (MAX_PROMPT_TEXT_CHARS * 3)
        prompt = build_handler_ranking_prompt(long_desc, ["a"])
        # Exactly MAX chars worth of x's, no more.
        assert prompt.count("x") == MAX_PROMPT_TEXT_CHARS

    def test_handles_empty_description(self) -> None:
        prompt = build_handler_ranking_prompt("", ["a"])
        assert "<task></task>" in prompt


# ---------------------------------------------------------------------------
# Crew.consensus field validator
# ---------------------------------------------------------------------------


class TestConsensusFieldValidator:
    def test_default_is_none(self) -> None:
        crew = Crew(agents=[_agent("a")], tasks=[], process=Process.sequential)
        assert crew.consensus is None

    def test_accepts_engine_instance(self) -> None:
        engine = MajorityVoteConsensus()
        crew = Crew(
            agents=[_agent("a")],
            tasks=[],
            process=Process.sequential,
            consensus=engine,
        )
        assert crew.consensus is engine

    def test_rejects_non_engine(self) -> None:
        """A non-string, non-engine value fails the structural Protocol check.

        Strings are now valid input form (resolved via ``discover_engines``),
        so the old "string -> aggregate-method missing" path is gone — the
        structural check fires only on real objects without ``aggregate``.
        """

        class _NotAnEngine:
            pass

        with pytest.raises(Exception, match="aggregate"):
            Crew(
                agents=[_agent("a")],
                tasks=[],
                process=Process.sequential,
                consensus=_NotAnEngine(),
            )


# ---------------------------------------------------------------------------
# Process.consensual end-to-end (with mocked agents and execution)
# ---------------------------------------------------------------------------


def _make_consensual_crew(
    agent_roles: list[str],
    task_description: str = "write something",
    task_agent: Agent | None = None,
) -> tuple[Crew, Task]:
    agents = [_agent(r) for r in agent_roles]
    task = Task(
        description=task_description,
        expected_output="something",
        agent=task_agent,
    )
    crew = Crew(agents=agents, tasks=[task], process=Process.consensual)
    return crew, task


class TestProcessConsensual:
    def test_unanimous_winner_assigned_to_task(self) -> None:
        crew, _task = _make_consensual_crew(["alice", "bob", "carol"])

        # Every voter ranks "alice" first among the *other* roles.
        # _rank_one prompts each agent to rank only the OTHER agents,
        # so we return a JSON array that always starts with "alice"
        # (when alice is in the eligible set) or "bob" otherwise.
        def _rank(sub: Task) -> str:
            voter = sub.agent.role  # type: ignore[union-attr]
            if voter == "alice":
                return json.dumps(["bob", "carol"])
            return json.dumps(["alice", "carol" if voter == "bob" else "bob"])

        captured: dict[str, Any] = {}

        def _execute(self_crew: Crew, tasks: list[Task]) -> CrewOutput:
            captured["winner"] = tasks[0].agent.role  # type: ignore[union-attr]
            return CrewOutput(
                raw="ok",
                tasks_output=[],
                token_usage=UsageMetrics(),
            )

        with (
            patch.object(Agent, "execute_task", side_effect=_rank),
            patch.object(Crew, "_execute_tasks", autospec=True, side_effect=_execute),
        ):
            crew.kickoff()

        assert captured["winner"] == "alice"

    def test_explicit_task_agent_is_not_overridden(self) -> None:
        pinned = _agent("alice")
        crew, _task = _make_consensual_crew(["alice", "bob"], task_agent=pinned)
        # Replace the existing agent list to keep references aligned.
        crew.agents = [pinned, _agent("bob")]

        captured: dict[str, Any] = {}

        def _execute(self_crew: Crew, tasks: list[Task]) -> CrewOutput:
            captured["winner"] = tasks[0].agent.role  # type: ignore[union-attr]
            return CrewOutput(raw="ok", tasks_output=[], token_usage=UsageMetrics())

        # Agent.execute_task should NEVER be called for ranking when the
        # task already has an agent — assert that by raising on any call.
        def _no_calls(_sub: Task) -> str:
            raise AssertionError("execute_task should not be called for pinned task")

        with (
            patch.object(Agent, "execute_task", side_effect=_no_calls),
            patch.object(Crew, "_execute_tasks", autospec=True, side_effect=_execute),
        ):
            crew.kickoff()

        assert captured["winner"] == "alice"

    def test_duplicate_roles_raises(self) -> None:
        crew, _task = _make_consensual_crew(["alice", "alice"])
        with pytest.raises(ValueError, match="unique agent roles"):
            crew.kickoff()

    def test_low_quorum_raises(self) -> None:
        crew, _task = _make_consensual_crew(["alice", "bob", "carol", "dave"])

        # Every voter fails — quorum will be 0/4, below the 0.5 threshold.
        def _fail(_sub: Task) -> str:
            raise RuntimeError("LLM exploded")

        with patch.object(Agent, "execute_task", side_effect=_fail):
            with pytest.raises(RuntimeError, match="valid handler rankings"):
                crew.kickoff()

    def test_custom_consensus_engine_is_used(self) -> None:
        crew, _task = _make_consensual_crew(["alice", "bob"])

        class AlwaysBob:
            def aggregate(self, candidates: Any, rankings: Any) -> str:
                return "bob"

        crew.consensus = AlwaysBob()

        def _rank(sub: Task) -> str:
            voter = sub.agent.role  # type: ignore[union-attr]
            return json.dumps(["bob"] if voter == "alice" else ["alice"])

        captured: dict[str, Any] = {}

        def _execute(self_crew: Crew, tasks: list[Task]) -> CrewOutput:
            captured["winner"] = tasks[0].agent.role  # type: ignore[union-attr]
            return CrewOutput(raw="ok", tasks_output=[], token_usage=UsageMetrics())

        with (
            patch.object(Agent, "execute_task", side_effect=_rank),
            patch.object(Crew, "_execute_tasks", autospec=True, side_effect=_execute),
        ):
            crew.kickoff()

        assert captured["winner"] == "bob"


# ---------------------------------------------------------------------------
# Plugin discovery + string shorthand
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_discovery_cache() -> None:
    """Clear the ``discover_engines`` LRU cache before every test in this module.

    Tests in this section monkey-patch entry points and the known-engine
    registry; the cache would silently keep stale results otherwise.
    """
    discover_engines.cache_clear()


def _make_entry_points_factory(
    *eps: object,
) -> Callable[[str], list[object]]:
    """Return a stub for ``importlib.metadata.entry_points`` that yields
    the given fakes only for the ``crewai.consensus_engines`` group.

    Centralised so tests don't repeat the lambda + group filter."""

    def _entry_points(group: str) -> list[object]:
        return list(eps) if group == "crewai.consensus_engines" else []

    return _entry_points


class TestDiscoverEngines:
    def test_built_in_majority_always_present(self) -> None:
        engines = discover_engines()
        assert engines["majority"] is MajorityVoteConsensus

    def test_known_fallback_resolves_when_module_importable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine in ``_KNOWN_ENGINE_IMPORT_PATHS`` is discovered by name
        when its canonical module is importable, even without an entry point."""
        import sys
        import types

        from crewai import consensus as consensus_module

        fake_mod = types.ModuleType("fake_plugin.engine")

        class _FakeEngine:
            def aggregate(
                self,
                candidates: Sequence[str],
                rankings: Mapping[str, Sequence[str]],
            ) -> str:
                return next(iter(rankings.values()))[0]

        fake_mod._FakeEngine = _FakeEngine  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fake_plugin", types.ModuleType("fake_plugin"))
        monkeypatch.setitem(sys.modules, "fake_plugin.engine", fake_mod)
        monkeypatch.setitem(
            consensus_module._KNOWN_ENGINE_IMPORT_PATHS,
            "fake",
            "fake_plugin.engine:_FakeEngine",
        )

        engines = discover_engines()
        assert engines["fake"] is _FakeEngine

    def test_known_fallback_skipped_when_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry in ``_KNOWN_ENGINE_IMPORT_PATHS`` whose module isn't
        importable is silently skipped (does not raise, does not appear)."""
        from crewai import consensus as consensus_module

        monkeypatch.setitem(
            consensus_module._KNOWN_ENGINE_IMPORT_PATHS,
            "definitely_not_installed",
            "no_such_module_xyz_123:Engine",
        )
        engines = discover_engines()
        assert "definitely_not_installed" not in engines

    def test_known_fallback_module_raising_non_import_error_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A fallback whose import raises something other than ``ImportError``
        (e.g. a misconfigured optional dep) must log a warning, not crash all
        of ``discover_engines``."""
        import logging

        from crewai import consensus as consensus_module

        # Patch ``importlib.import_module`` to raise non-ImportError for our
        # synthetic path; real imports still work via the saved reference.
        real_import = consensus_module.importlib.import_module

        def fake_import(name: str) -> object:
            if name == "broken_plugin":
                raise RuntimeError("misconfigured optional dep")
            return real_import(name)

        monkeypatch.setattr(consensus_module.importlib, "import_module", fake_import)
        monkeypatch.setitem(
            consensus_module._KNOWN_ENGINE_IMPORT_PATHS,
            "broken",
            "broken_plugin:Engine",
        )

        with caplog.at_level(logging.WARNING, logger="crewai.consensus"):
            engines = discover_engines()

        assert "broken" not in engines
        assert any(
            "broken" in r.message and "raised at import" in r.message
            for r in caplog.records
        )

    def test_known_fallback_missing_attribute_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A fallback whose module imports cleanly but lacks the named class
        must log, not silently drop."""
        import logging
        import sys
        import types

        from crewai import consensus as consensus_module

        partial_mod = types.ModuleType("partial_plugin")
        # Intentionally no ``Engine`` attribute.
        monkeypatch.setitem(sys.modules, "partial_plugin", partial_mod)
        monkeypatch.setitem(
            consensus_module._KNOWN_ENGINE_IMPORT_PATHS,
            "partial",
            "partial_plugin:Engine",
        )

        with caplog.at_level(logging.WARNING, logger="crewai.consensus"):
            engines = discover_engines()

        assert "partial" not in engines
        assert any(
            "partial" in r.message and "no attribute" in r.message
            for r in caplog.records
        )

    def test_entry_point_wins_over_known_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry-point registration overrides the hard-coded fallback for
        the same name."""
        from crewai import consensus as consensus_module

        class _FromEntryPoint:
            def aggregate(
                self,
                candidates: Sequence[str],
                rankings: Mapping[str, Sequence[str]],
            ) -> str:
                return "x"

        class _FakeEntryPoint:
            name = "snowveil"

            def load(self) -> type:
                return _FromEntryPoint

        monkeypatch.setattr(
            consensus_module.importlib.metadata,
            "entry_points",
            _make_entry_points_factory(_FakeEntryPoint()),
        )
        engines = discover_engines()
        assert engines["snowveil"] is _FromEntryPoint

    def test_failed_entry_point_load_logs_warning_and_skips(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A plugin that errors during ``ep.load()`` is logged and skipped —
        a broken third-party engine must not crash an unrelated crew."""
        import logging

        from crewai import consensus as consensus_module

        class _BrokenEntryPoint:
            name = "broken"

            def load(self) -> type:
                raise RuntimeError("plugin import blew up")

        monkeypatch.setattr(
            consensus_module.importlib.metadata,
            "entry_points",
            _make_entry_points_factory(_BrokenEntryPoint()),
        )
        with caplog.at_level(logging.WARNING, logger="crewai.consensus"):
            engines = discover_engines()
        assert "broken" not in engines
        assert any("broken" in r.message for r in caplog.records)

    def test_entry_point_returning_non_class_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An entry point that returns a non-class value (an instance, a
        function, a module) must be rejected at discovery time. Otherwise
        ``engines[name]()`` later produces a confusing ``not callable`` error."""
        import logging

        from crewai import consensus as consensus_module

        class _NonClassEntryPoint:
            name = "weird"

            def load(self) -> object:
                return "this is a string, not a class"

        monkeypatch.setattr(
            consensus_module.importlib.metadata,
            "entry_points",
            _make_entry_points_factory(_NonClassEntryPoint()),
        )
        with caplog.at_level(logging.WARNING, logger="crewai.consensus"):
            engines = discover_engines()
        assert "weird" not in engines
        assert any(
            "weird" in r.message and "not a class" in r.message for r in caplog.records
        )

    def test_duplicate_entry_point_names_logs_collision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Two entry points sharing a name produce a collision warning. Last
        registration wins (dict insertion order); the warning surfaces the
        shadowing for debugging."""
        import logging

        from crewai import consensus as consensus_module

        class _First:
            def aggregate(self, c, r):  # type: ignore[no-untyped-def]
                return "first"

        class _Second:
            def aggregate(self, c, r):  # type: ignore[no-untyped-def]
                return "second"

        class _EpA:
            name = "dupe"

            def load(self) -> type:
                return _First

        class _EpB:
            name = "dupe"

            def load(self) -> type:
                return _Second

        monkeypatch.setattr(
            consensus_module.importlib.metadata,
            "entry_points",
            _make_entry_points_factory(_EpA(), _EpB()),
        )
        with caplog.at_level(logging.WARNING, logger="crewai.consensus"):
            engines = discover_engines()
        assert engines["dupe"] is _Second  # last write wins
        assert any(
            "dupe" in r.message and "multiple" in r.message for r in caplog.records
        )

    def test_two_named_plugins_coexist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distinct plugin names register independently."""
        from crewai import consensus as consensus_module

        class _PluginA:
            def aggregate(self, c, r):  # type: ignore[no-untyped-def]
                return "a"

        class _PluginB:
            def aggregate(self, c, r):  # type: ignore[no-untyped-def]
                return "b"

        class _EpA:
            name = "plugin_a"

            def load(self) -> type:
                return _PluginA

        class _EpB:
            name = "plugin_b"

            def load(self) -> type:
                return _PluginB

        monkeypatch.setattr(
            consensus_module.importlib.metadata,
            "entry_points",
            _make_entry_points_factory(_EpA(), _EpB()),
        )
        engines = discover_engines()
        assert engines["plugin_a"] is _PluginA
        assert engines["plugin_b"] is _PluginB
        assert engines["majority"] is MajorityVoteConsensus  # built-in still present

    def test_cache_returns_same_dict_until_cleared(self) -> None:
        """Repeated calls return the same object (cache hit). After
        ``cache_clear()``, a fresh dict is built."""
        first = discover_engines()
        second = discover_engines()
        assert first is second
        discover_engines.cache_clear()
        third = discover_engines()
        assert third is not first


class TestConsensusFieldStringShorthand:
    def test_string_resolves_to_default_majority_instance(self) -> None:
        """``Crew(consensus=\"majority\")`` produces a working
        ``MajorityVoteConsensus`` instance."""
        crew = Crew(
            agents=[_agent("alice"), _agent("bob")],
            tasks=[Task(description="t", expected_output="o")],
            process=Process.consensual,
            consensus="majority",
        )
        assert isinstance(crew.consensus, MajorityVoteConsensus)

    def test_unknown_string_raises_with_helpful_error(self) -> None:
        """Passing an unknown engine name lists the installed alternatives."""
        with pytest.raises(ValueError, match="unknown consensus engine name"):
            Crew(
                agents=[_agent("alice"), _agent("bob")],
                tasks=[Task(description="t", expected_output="o")],
                process=Process.consensual,
                consensus="not_a_real_engine_xyz",
            )

    def test_empty_string_raises_dedicated_error(self) -> None:
        """``consensus=\"\"`` produces a clearer message than the unknown-name
        error (treats empty string as a separate misuse)."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Crew(
                agents=[_agent("alice"), _agent("bob")],
                tasks=[Task(description="t", expected_output="o")],
                process=Process.consensual,
                consensus="",
            )

    def test_instance_still_accepted(self) -> None:
        """Passing an instance directly still works (string is shorthand,
        not a replacement)."""
        crew = Crew(
            agents=[_agent("alice"), _agent("bob")],
            tasks=[Task(description="t", expected_output="o")],
            process=Process.consensual,
            consensus=MajorityVoteConsensus(),
        )
        assert isinstance(crew.consensus, MajorityVoteConsensus)

    def test_instance_path_does_not_call_discover_engines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passing an instance must not trigger discovery — ``discover_engines``
        is only consulted for string lookups. Important once caching becomes
        a perf-sensitive concern."""
        import crewai.crew as crew_module

        call_count = {"n": 0}
        real_discover = crew_module.discover_engines

        def counting_discover() -> dict[str, type]:
            call_count["n"] += 1
            return real_discover()

        monkeypatch.setattr(crew_module, "discover_engines", counting_discover)

        Crew(
            agents=[_agent("alice"), _agent("bob")],
            tasks=[Task(description="t", expected_output="o")],
            process=Process.consensual,
            consensus=MajorityVoteConsensus(),
        )
        assert call_count["n"] == 0

"""Regression tests for issue #5736: parallel agent executors must be consistent.

The repository has historically carried two parallel ``BaseAgentExecutor``
subclasses — :class:`crewai.agents.crew_agent_executor.CrewAgentExecutor` and
:class:`crewai.experimental.agent_executor.AgentExecutor`. The issue reported
two concrete bugs caused by the split:

1. ``Agent.kickoff()`` silently ignored ``Agent.executor_class`` and always
   instantiated ``AgentExecutor``, so the user-visible default
   (``CrewAgentExecutor``) did not match the executor that ``kickoff()``
   actually used.
2. The ``handle_reasoning()`` step was gated on a literal class identity check
   (``self.executor_class is not AgentExecutor``), so reasoning was silently
   skipped depending on the executor and could not be controlled via a
   capability flag.

These tests pin down the new contract:

* Capability flags (``supports_internal_planning`` / ``supports_kickoff``) are
  the single source of truth for executor differences.
* ``Agent.kickoff()`` honors ``executor_class`` when it is kickoff-capable and
  otherwise emits a ``DeprecationWarning`` instead of silently overriding.
* ``Agent.execute_task()`` instantiates ``self.executor_class`` (not a
  hard-coded class).
* ``handle_reasoning()`` is gated on the capability flag, not class identity.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal
import warnings
from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Task
from crewai.agents.agent_builder.base_agent_executor import BaseAgentExecutor
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.experimental.agent_executor import AgentExecutor
from crewai.llms.base_llm import BaseLLM


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM that satisfies BaseLLM's protocol."""
    llm = MagicMock(spec=BaseLLM)
    llm.supports_stop_words.return_value = True
    llm.stop = []
    llm.model = "gpt-4o-mini"
    return llm


@pytest.fixture
def test_agent(mock_llm: MagicMock) -> Agent:
    """A plain Agent that uses default ``executor_class``."""
    return Agent(
        role="Test Agent",
        goal="Verify executor consistency",
        backstory="An agent used by issue #5736 regression tests.",
        llm=mock_llm,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


class TestExecutorCapabilityFlags:
    """``supports_internal_planning`` / ``supports_kickoff`` advertise the
    differences between executors so callers don't have to ``isinstance``-check
    against a specific class."""

    def test_base_executor_has_no_capabilities(self) -> None:
        assert BaseAgentExecutor.supports_internal_planning is False
        assert BaseAgentExecutor.supports_kickoff is False

    def test_crew_agent_executor_has_no_capabilities(self) -> None:
        """The legacy executor neither plans internally nor handles kickoff."""
        assert CrewAgentExecutor.supports_internal_planning is False
        assert CrewAgentExecutor.supports_kickoff is False

    def test_agent_executor_advertises_full_capabilities(self) -> None:
        """The Flow-based executor implements both plan-and-execute and kickoff."""
        assert AgentExecutor.supports_internal_planning is True
        assert AgentExecutor.supports_kickoff is True


# ---------------------------------------------------------------------------
# Agent.kickoff() honors executor_class
# ---------------------------------------------------------------------------


class TestKickoffRespectsExecutorClass:
    """Issue #5736 part 1: ``Agent.kickoff()`` must honor ``executor_class``."""

    def test_kickoff_uses_explicit_agent_executor_without_warning(
        self, mock_llm: MagicMock
    ) -> None:
        """When the user explicitly opts into ``AgentExecutor``, no warning fires."""
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            llm=mock_llm,
            executor_class=AgentExecutor,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            executor_cls = agent._resolve_kickoff_executor_class()

        assert executor_cls is AgentExecutor
        kickoff_warnings = [
            w for w in caught if "Agent.kickoff()" in str(w.message)
        ]
        assert kickoff_warnings == [], (
            f"Did not expect any kickoff warning, got: {[str(w.message) for w in kickoff_warnings]}"
        )

    def test_kickoff_uses_kickoff_capable_subclass_of_agent_executor(
        self, mock_llm: MagicMock
    ) -> None:
        """A user-defined subclass that opts into kickoff is honored verbatim
        (rather than being silently replaced with ``AgentExecutor``)."""

        class CustomAgentExecutor(AgentExecutor):
            executor_type: Literal["custom"] = "custom"  # type: ignore[assignment]
            supports_kickoff: ClassVar[bool] = True
            supports_internal_planning: ClassVar[bool] = True

        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            llm=mock_llm,
            executor_class=CustomAgentExecutor,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            executor_cls = agent._resolve_kickoff_executor_class()

        assert executor_cls is CustomAgentExecutor
        kickoff_warnings = [
            w for w in caught if "Agent.kickoff()" in str(w.message)
        ]
        assert kickoff_warnings == []

    def test_kickoff_warns_and_falls_back_when_executor_class_is_not_kickoff_capable(
        self, mock_llm: MagicMock
    ) -> None:
        """Default-configured agents (``executor_class=CrewAgentExecutor``) used
        to silently flip to ``AgentExecutor`` on ``kickoff()``. The fix keeps
        the fallback for backwards compatibility but emits a
        ``DeprecationWarning`` so the silent override is now visible."""
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            llm=mock_llm,
            executor_class=CrewAgentExecutor,  # the default
        )

        with pytest.warns(DeprecationWarning, match=r"Agent\.kickoff\(\)"):
            executor_cls = agent._resolve_kickoff_executor_class()

        assert executor_cls is AgentExecutor

    def test_kickoff_fallback_warning_mentions_explicit_opt_in(
        self, mock_llm: MagicMock
    ) -> None:
        """The deprecation message should tell users how to silence it — by
        setting ``executor_class=AgentExecutor`` explicitly."""
        agent = Agent(
            role="r",
            goal="g",
            backstory="b",
            llm=mock_llm,
            executor_class=CrewAgentExecutor,
        )

        with pytest.warns(DeprecationWarning) as record:
            agent._resolve_kickoff_executor_class()

        assert any(
            "executor_class=AgentExecutor" in str(w.message) for w in record
        )

    def test_kickoff_constructs_executor_via_resolve_helper(
        self, test_agent: Agent
    ) -> None:
        """``_prepare_kickoff()`` must route through ``_resolve_kickoff_executor_class``
        rather than instantiating ``AgentExecutor`` directly. This guards against
        regressing back to the issue #5736 hardcoded-class behavior."""
        sentinel = MagicMock(name="sentinel_executor_cls")
        sentinel.return_value = MagicMock(name="sentinel_executor")

        with patch.object(
            test_agent,
            "_resolve_kickoff_executor_class",
            return_value=sentinel,
        ) as mock_resolve, patch.object(
            test_agent, "_build_execution_prompt"
        ) as mock_build_prompt:
            mock_build_prompt.return_value = (
                {"system": "sys"},
                ["Observation:"],
                None,
            )
            test_agent._prepare_kickoff(messages="hello")

        mock_resolve.assert_called_once_with()
        sentinel.assert_called_once()


# ---------------------------------------------------------------------------
# Agent.execute_task() honors executor_class (sanity check; pre-existing
# behavior we don't want to regress)
# ---------------------------------------------------------------------------


class TestExecuteTaskRespectsExecutorClass:
    """``Agent.execute_task()`` already routed through ``self.executor_class``
    — these tests pin that contract so future refactors keep both entry points
    honoring the user's executor choice."""

    def test_execute_task_instantiates_self_executor_class(
        self, test_agent: Agent
    ) -> None:
        """The non-resuming branch of ``Agent.create_agent_executor`` should
        call ``self.executor_class(...)``. Stubbing ``executor_class`` with a
        factory lets us observe the instantiation without running the LLM
        loop."""
        captured: dict[str, Any] = {}

        def fake_factory(*args: Any, **kwargs: Any) -> MagicMock:
            captured["called"] = True
            captured["kwargs"] = kwargs
            return MagicMock(spec=CrewAgentExecutor)

        task = Task(
            description="d",
            expected_output="e",
            agent=test_agent,
        )
        test_agent.agent_executor = None
        test_agent.executor_class = fake_factory  # type: ignore[assignment]

        with patch.object(test_agent, "_build_execution_prompt") as mock_bp:
            mock_bp.return_value = ({"system": "s"}, ["Obs:"], None)
            test_agent.create_agent_executor(task=task)

        assert captured.get("called") is True
        # The agent passed itself to its executor.
        assert captured["kwargs"].get("agent") is test_agent
        assert captured["kwargs"].get("task") is task


# ---------------------------------------------------------------------------
# handle_reasoning gate: capability flag, not class identity
# ---------------------------------------------------------------------------


class TestReasoningGateUsesCapabilityFlag:
    """Issue #5736 part 2: the reasoning gate must consult
    ``supports_internal_planning`` rather than testing class identity. This
    way a ``CrewAgentExecutor`` subclass that opts into internal planning is
    treated correctly — and so is an ``AgentExecutor`` subclass that doesn't."""

    def test_reasoning_runs_for_executor_without_internal_planning(
        self, test_agent: Agent
    ) -> None:
        """Default ``executor_class`` is ``CrewAgentExecutor`` which does NOT
        plan internally, so the legacy ``handle_reasoning`` step must run."""
        task = Task(description="d", expected_output="e", agent=test_agent)

        assert test_agent.executor_class is CrewAgentExecutor
        assert (
            getattr(
                test_agent.executor_class, "supports_internal_planning", False
            )
            is False
        )

        with patch(
            "crewai.agent.core.handle_reasoning"
        ) as mock_handle_reasoning, patch.object(
            test_agent, "_inject_date_to_task"
        ), patch.object(
            test_agent, "_retrieve_memory_context", return_value="prompt"
        ):
            test_agent._prepare_task_execution(task=task, context=None)

        mock_handle_reasoning.assert_called_once_with(test_agent, task)

    def test_reasoning_skipped_for_executor_with_internal_planning(
        self, test_agent: Agent
    ) -> None:
        """``AgentExecutor.supports_internal_planning is True``, so the agent
        should not invoke the external ``handle_reasoning`` step (the executor
        runs its own ``generate_plan`` flow instead)."""
        test_agent.executor_class = AgentExecutor  # type: ignore[assignment]
        task = Task(description="d", expected_output="e", agent=test_agent)

        assert (
            getattr(
                test_agent.executor_class, "supports_internal_planning", False
            )
            is True
        )

        with patch(
            "crewai.agent.core.handle_reasoning"
        ) as mock_handle_reasoning, patch.object(
            test_agent, "_inject_date_to_task"
        ), patch.object(
            test_agent, "_retrieve_memory_context", return_value="prompt"
        ):
            test_agent._prepare_task_execution(task=task, context=None)

        mock_handle_reasoning.assert_not_called()

    def test_reasoning_gate_does_not_use_isinstance_or_class_identity(
        self, test_agent: Agent
    ) -> None:
        """A ``CrewAgentExecutor`` subclass that opts into internal planning
        (``supports_internal_planning = True``) must skip the legacy reasoning
        step. This is the key behavioral promise of replacing the old
        ``self.executor_class is not AgentExecutor`` check with a flag-based
        check."""

        class PlanningCrewExecutor(CrewAgentExecutor):
            executor_type: Literal["crew_planning"] = "crew_planning"  # type: ignore[assignment]
            supports_internal_planning: ClassVar[bool] = True

        test_agent.executor_class = PlanningCrewExecutor  # type: ignore[assignment]
        task = Task(description="d", expected_output="e", agent=test_agent)

        with patch(
            "crewai.agent.core.handle_reasoning"
        ) as mock_handle_reasoning, patch.object(
            test_agent, "_inject_date_to_task"
        ), patch.object(
            test_agent, "_retrieve_memory_context", return_value="prompt"
        ):
            test_agent._prepare_task_execution(task=task, context=None)

        mock_handle_reasoning.assert_not_called()

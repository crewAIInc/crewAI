"""Regression tests: a second kickoff of a checkpointed crew or agent runs fresh.

Saving a checkpoint syncs the live object's ``_kickoff_event_id`` into its
public ``checkpoint_kickoff_event_id`` (``_sync_checkpoint_fields``), and
nothing cleared it after the run. The next ``kickoff()`` on the same instance
read that stale pointer as a resume request:

- a crew skipped every task (each still held the previous run's output) and
  returned the previous run's result without calling the LLM, ignoring the
  new inputs;
- an agent emitted no ``LiteAgentExecutionStartedEvent`` and reused the
  previous run's kickoff id.

Only an instance restored by ``from_checkpoint`` should resume.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import LiteAgentExecutionStartedEvent
from crewai.events.types.crew_events import CrewKickoffStartedEvent
from crewai.llms.base_llm import BaseLLM
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.checkpoint_listener import _ensure_handlers_registered
import pytest


@pytest.fixture(autouse=True)
def _checkpoint_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    # The checkpoint listener registers once per process, but conftest wipes
    # the bus after every test; re-register it so checkpoints are really
    # written (otherwise these tests would pass without exercising the bug).
    # Handlers added here and in the tests are cleared by that same fixture.
    monkeypatch.setattr(
        "crewai.state.checkpoint_listener._handlers_registered", False
    )
    _ensure_handlers_registered()


class _CountingLLM(BaseLLM):
    """Stub LLM that numbers its final answers, so stale results are visible."""

    def __init__(self) -> None:
        super().__init__(model="stub")
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        self._calls += 1
        return f"Final Answer: answer-{self._calls}"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError


def _checkpoint(location: str) -> CheckpointConfig:
    return CheckpointConfig(location=location, on_events=["*"])


def _latest_checkpoint(location: str) -> str:
    files = sorted(Path(location).rglob("*.json"), key=lambda p: p.stat().st_mtime)
    assert files, "no checkpoint was written"
    return str(files[-1])


class TestCrewKickoffTwiceWithCheckpointing:
    def test_second_kickoff_executes_with_new_inputs(self) -> None:
        llm = _CountingLLM()
        agent = Agent(role="r", goal="g", backstory="b", llm=llm)
        task = Task(description="About {topic}", expected_output="x", agent=agent)
        started: list[str] = []

        with tempfile.TemporaryDirectory() as d:

            @crewai_event_bus.on(CrewKickoffStartedEvent)
            def _on_started(_: Any, event: CrewKickoffStartedEvent) -> None:
                started.append(event.event_id)

            crew = Crew(agents=[agent], tasks=[task], checkpoint=_checkpoint(d))
            first = crew.kickoff(inputs={"topic": "cats"})
            second = crew.kickoff(inputs={"topic": "dogs"})
            crewai_event_bus.flush()
            _latest_checkpoint(d)  # guard: checkpointing really ran

        assert first.raw == "answer-1"
        assert second.raw == "answer-2"
        assert llm.calls == 2
        assert len(started) == 2
        assert started[0] != started[1]

    def test_restored_crew_resumes_then_runs_fresh(self) -> None:
        llm = _CountingLLM()
        agent = Agent(role="r", goal="g", backstory="b", llm=llm)
        task = Task(description="d", expected_output="x", agent=agent)

        with tempfile.TemporaryDirectory() as d:
            Crew(agents=[agent], tasks=[task], checkpoint=_checkpoint(d)).kickoff()
            crewai_event_bus.flush()

            restored = Crew.from_checkpoint(
                CheckpointConfig(restore_from=_latest_checkpoint(d))
            )
            restored.agents[0].llm = llm

            # The restored crew resumes: its only task is complete, so
            # nothing re-executes.
            calls_before_resume = llm.calls
            restored.kickoff()
            assert llm.calls == calls_before_resume

            # Its next kickoff is a new run, not another resume.
            again = restored.kickoff()
            crewai_event_bus.flush()

        assert llm.calls == calls_before_resume + 1
        assert again.raw == f"answer-{llm.calls}"


class TestAgentKickoffTwiceWithCheckpointing:
    def test_second_kickoff_emits_its_own_started_event(self) -> None:
        started: list[str] = []

        with tempfile.TemporaryDirectory() as d:

            @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
            def _on_started(_: Any, event: LiteAgentExecutionStartedEvent) -> None:
                started.append(event.event_id)

            agent = Agent(
                role="r",
                goal="g",
                backstory="b",
                llm=_CountingLLM(),
                checkpoint=_checkpoint(d),
            )
            agent.kickoff("first")
            first_id = agent._kickoff_event_id
            agent.kickoff("second")
            second_id = agent._kickoff_event_id
            crewai_event_bus.flush()
            _latest_checkpoint(d)  # guard: checkpointing really ran

        assert len(started) == 2
        assert first_id != second_id
        assert second_id == started[1]

    def test_copy_drops_checkpoint_kickoff_event_id(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b", llm=_CountingLLM())
        agent.checkpoint_kickoff_event_id = "evt-1"

        assert agent.copy().checkpoint_kickoff_event_id is None

    def test_restored_agent_resumes_once(self) -> None:
        agent = Agent(role="r", goal="g", backstory="b", llm=_CountingLLM())
        agent.checkpoint_kickoff_event_id = "evt-restored"

        # A fresh (or ``from_checkpoint``-restored) agent resumes the pointer.
        assert agent._take_resume_kickoff_event_id() == "evt-restored"
        # Once it has kicked off, a pointer re-synced by a checkpoint save is
        # stale and must not resume.
        agent.checkpoint_kickoff_event_id = "evt-restored"
        assert agent._take_resume_kickoff_event_id() is None

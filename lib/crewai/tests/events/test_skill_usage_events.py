"""Tests for runtime skill usage events.

Discovery/load/activation events cover setup. `SkillUsedEvent` is the runtime
signal: it fires whenever a skill's context is injected into an agent's prompt
for a task, so observability can attribute usage to an agent and task.
"""

from pathlib import Path

import pytest

from crewai import Agent, Task
from crewai.events import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent
from crewai.skills.loader import activate_skill, discover_skills
from crewai.skills.tool import LoadSkillTool


# Handlers run on the event bus thread pool, so tests must flush before
# asserting — otherwise assertions race the handler and pass/fail on timing.


def _create_skill_dir(parent: Path, name: str, body: str = "Body.") -> Path:
    """Create a skill directory containing a minimal SKILL.md."""
    skill_dir = parent / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n---\n{body}"
    )
    return skill_dir


def _agent_with_skills(search_path: Path) -> Agent:
    """Create an agent with explicitly activated, always-on skills."""
    skills = [activate_skill(skill) for skill in discover_skills(search_path)]
    return Agent(
        role="Analyst",
        goal="Analyze things",
        backstory="Experienced",
        skills=skills,
        llm="gpt-4o-mini",
    )


def _task_for(agent: Agent) -> Task:
    return Task(description="Do the thing", expected_output="text", agent=agent)


class TestSkillUsedEvent:
    def test_metadata_skill_emits_only_after_runtime_selection(
        self, tmp_path: Path
    ) -> None:
        _create_skill_dir(tmp_path, "alpha")
        agent = Agent(
            role="Analyst",
            goal="Analyze things",
            backstory="Experienced",
            skills=[tmp_path],
            llm="gpt-4o-mini",
        )
        task = _task_for(agent)
        agent.create_agent_executor(task=task)
        assert agent.agent_executor is not None
        loader = next(
            tool
            for tool in agent.agent_executor.original_tools
            if isinstance(tool, LoadSkillTool)
        )

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._emit_skill_usage(task)
            assert crewai_event_bus.flush(timeout=10)
            assert received == []

            loader.run(skill_name="alpha")
            assert crewai_event_bus.flush(timeout=10)

        assert [event.skill_name for event in received] == ["alpha"]
        assert received[0].task_id == str(task.id)

    def test_emits_one_event_per_skill_with_agent_and_task_attribution(
        self, tmp_path: Path
    ) -> None:
        _create_skill_dir(tmp_path, "alpha")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._finalize_task_prompt("PROMPT", [], task)
            assert crewai_event_bus.flush(timeout=10)

        assert len(received) == 1
        event = received[0]
        assert event.type == "skill_used"
        assert event.skill_name == "alpha"
        assert event.skill_path is not None
        # Attribution is what makes the event useful in traces.
        assert event.agent_role == "Analyst"
        assert event.agent_id == str(agent.id)
        assert event.task_id == str(task.id)

    def test_emits_for_every_skill(self, tmp_path: Path) -> None:
        _create_skill_dir(tmp_path, "alpha")
        _create_skill_dir(tmp_path, "beta")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._finalize_task_prompt("PROMPT", [], task)
            assert crewai_event_bus.flush(timeout=10)

        assert sorted(e.skill_name for e in received) == ["alpha", "beta"]

    def test_re_emits_on_each_execution(self, tmp_path: Path) -> None:
        """Usage is per-execution, unlike activation which is idempotent."""
        _create_skill_dir(tmp_path, "alpha")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._finalize_task_prompt("PROMPT", [], task)
            agent._finalize_task_prompt("PROMPT", [], task)
            assert crewai_event_bus.flush(timeout=10)

        assert len(received) == 2

    def test_reports_disclosure_level(self, tmp_path: Path) -> None:
        """Explicitly activated skills report INSTRUCTIONS level."""
        _create_skill_dir(tmp_path, "alpha")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._finalize_task_prompt("PROMPT", [], task)
            assert crewai_event_bus.flush(timeout=10)

        assert received[0].disclosure_level >= 2

    def test_agent_without_skills_emits_nothing(self) -> None:
        agent = Agent(
            role="Analyst",
            goal="Analyze things",
            backstory="Experienced",
            llm="gpt-4o-mini",
        )
        task = _task_for(agent)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent._finalize_task_prompt("PROMPT", [], task)
            assert crewai_event_bus.flush(timeout=10)

        assert received == []

    def test_event_is_exported_from_events_package(self) -> None:
        from crewai.events import SkillUsedEvent as Exported

        assert Exported is SkillUsedEvent


class TestSkillUsedEventThroughExecution:
    """Coverage through the public execution entry points.

    The helper-level tests above pin per-skill details; these prove the event
    actually reaches the bus during a real `execute_task` / `aexecute_task`
    run. The agent executor is stubbed so no LLM call is made.
    """

    def test_emitted_during_execute_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _create_skill_dir(tmp_path, "alpha")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)
        monkeypatch.setattr(
            Agent, "_execute_without_timeout", lambda self, prompt, task: "done"
        )

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            agent.execute_task(task)
            assert crewai_event_bus.flush(timeout=10)

        assert [e.skill_name for e in received] == ["alpha"]
        assert received[0].task_id == str(task.id)

    @pytest.mark.asyncio
    async def test_emitted_during_aexecute_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _create_skill_dir(tmp_path, "alpha")
        agent = _agent_with_skills(tmp_path)
        task = _task_for(agent)

        async def _fake_execute(self, prompt: str, task: Task) -> str:  # noqa: ANN001
            return "done"

        monkeypatch.setattr(Agent, "_aexecute_without_timeout", _fake_execute)

        received: list[SkillUsedEvent] = []
        with crewai_event_bus.scoped_handlers():

            @crewai_event_bus.on(SkillUsedEvent)
            def _handler(source, event: SkillUsedEvent) -> None:  # noqa: ARG001
                received.append(event)

            await agent.aexecute_task(task)
            assert crewai_event_bus.flush(timeout=10)

        assert [e.skill_name for e in received] == ["alpha"]
        assert received[0].task_id == str(task.id)

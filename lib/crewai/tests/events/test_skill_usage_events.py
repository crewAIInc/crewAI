"""Tests for runtime skill usage events.

Discovery/load/activation events cover setup. `SkillUsedEvent` is the runtime
signal: it fires whenever a skill's context is injected into an agent's prompt
for a task, so observability can attribute usage to an agent and task.
"""

from pathlib import Path

from crewai import Agent, Task
from crewai.events import crewai_event_bus
from crewai.events.types.skill_events import SkillUsedEvent


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
    # discover_skills scans the SUBdirectories of the given path.
    return Agent(
        role="Analyst",
        goal="Analyze things",
        backstory="Experienced",
        skills=[str(search_path)],
        llm="gpt-4o-mini",
    )


def _task_for(agent: Agent) -> Task:
    return Task(description="Do the thing", expected_output="text", agent=agent)


class TestSkillUsedEvent:
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
        assert event.agent_id
        assert event.task_id

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
        """Path-loaded skills are activated, so they report INSTRUCTIONS level."""
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

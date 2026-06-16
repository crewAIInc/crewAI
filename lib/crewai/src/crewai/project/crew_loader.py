"""Load crew definitions from JSON/JSONC files and produce Crew instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crewai.project.crew_definition import CrewDefinition
from crewai.project.json_loader import (
    JSONAgentDefinition,
    JSONCrewProject,
    JSONProjectError,
    JSONProjectValidationError,
    _AgentDefinitionSource,
    _crew_kwargs_from_definition,
    _load_json_crew_project_definition,
    _task_class_from_definition,
    _task_kwargs_from_definition,
    load_json_crew_project,
)


def load_crew(
    source: Path | str,
    agents_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a ``Crew`` from a JSON/JSONC definition file.

    The definition file describes the crew's agents, tasks, process type, and
    default inputs.  Agent definitions are resolved from individual
    ``<name>.jsonc`` / ``<name>.json`` files inside an ``agents/`` directory.
    """
    crew_path = Path(source)
    project = load_json_crew_project(crew_path, agents_dir=agents_dir)
    return _load_crew_project(project, project_root=crew_path.parent)


def load_crew_from_definition(
    definition: CrewDefinition | dict[str, Any],
    *,
    source: str | Path = "<inline crew>",
    project_root: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a ``Crew`` from an in-memory JSON/YAML crew definition."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    source_label = str(source)
    crew_definition = (
        definition
        if isinstance(definition, CrewDefinition)
        else CrewDefinition.model_validate(definition)
    )
    definition_data = crew_definition.model_dump(mode="python", exclude_none=True)
    project = _crew_project_from_definition(
        definition_data,
        source=source_label,
        project_root=root,
    )
    return _load_crew_project(project, project_root=root)


def _crew_project_from_definition(
    definition: dict[str, Any],
    *,
    source: str,
    project_root: Path,
) -> JSONCrewProject:
    agent_bodies: dict[str, Any] = definition["agents"]
    agent_names = list(agent_bodies)
    manager_agent = definition.get("manager_agent")
    if isinstance(manager_agent, str):
        agent_names = [name for name in agent_names if name != manager_agent]

    def load_agent_definition_source(agent_name: str) -> _AgentDefinitionSource | None:
        body = agent_bodies.get(agent_name)
        if body is None:
            return None
        return body, f"{source}: agents.{agent_name}"

    return _load_json_crew_project_definition(
        {**definition, "agents": agent_names},
        source=source,
        agents_dir=project_root / "agents",
        project_root=project_root,
        load_agent_definition_source=load_agent_definition_source,
        missing_agent_hint=None,
        collect_errors=False,
    )


def _load_crew_project(
    project: JSONCrewProject,
    *,
    project_root: Path,
) -> tuple[Any, dict[str, Any]]:
    from crewai import Crew, Task

    source_label = str(project.crew_path)

    def build_agent(agent_def: JSONAgentDefinition) -> Any:
        try:
            return agent_def.agent_class(**agent_def.kwargs)
        except ValidationError as exc:
            raise JSONProjectError(
                f"{agent_def.path}: validation failed: {exc}"
            ) from exc
        except Exception as exc:
            raise JSONProjectError(
                f"{agent_def.path}: failed to load agent: {exc}"
            ) from exc

    agents_map: dict[str, Any] = {}
    for name, agent_def in project.agents.items():
        agents_map[name] = build_agent(agent_def)

    tasks_list: list[Task] = []
    task_name_map: dict[str, Task] = {}

    for index, task_defn in enumerate(project.task_definitions):
        task_source = f"{source_label}: tasks[{index}]"
        task_class = _task_class_from_definition(task_defn, f"{task_source}: type")
        task_kwargs = _task_kwargs_from_definition(
            task_defn,
            agents_map=agents_map,
            task_name_map=task_name_map,
            source=task_source,
            project_root=project_root,
        )
        try:
            task = task_class(**task_kwargs)
        except ValidationError as exc:
            raise JSONProjectError(f"{task_source}: validation failed: {exc}") from exc
        except Exception as exc:
            raise JSONProjectError(
                f"{task_source}: failed to load task: {exc}"
            ) from exc

        tasks_list.append(task)
        task_name = task_defn.get("name")
        if isinstance(task_name, str) and task_name:
            task_name_map[task_name] = task

    crew_kwargs = _crew_kwargs_from_definition(
        project.definition,
        agents=[agents_map[name] for name in project.agent_names],
        tasks=tasks_list,
        agents_map=agents_map,
        source=source_label,
    )

    try:
        crew = Crew(**crew_kwargs)
    except ValidationError as exc:
        raise JSONProjectError(f"{source_label}: validation failed: {exc}") from exc
    except JSONProjectValidationError:
        raise
    except Exception as exc:
        raise JSONProjectError(f"{source_label}: failed to load crew: {exc}") from exc

    return crew, project.definition.get("inputs", {})


def load_crew_and_kickoff(
    crew_path: Path | str,
    input_overrides: dict[str, Any] | None = None,
) -> Any:
    """Convenience function: load a crew and immediately kick it off."""
    crew, default_inputs = load_crew(crew_path)

    merged_inputs = {**default_inputs}
    if input_overrides:
        merged_inputs.update(input_overrides)

    return crew.kickoff(inputs=merged_inputs)

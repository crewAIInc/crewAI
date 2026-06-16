"""Load crew definitions from JSON/JSONC files and produce Crew instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crewai.project.json_loader import (
    JSONProjectError,
    JSONProjectValidationError,
    _crew_kwargs_from_definition,
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
    from crewai import Crew, Task

    crew_path = Path(source)
    project = load_json_crew_project(crew_path, agents_dir=agents_dir)

    def build_agent(agent_def: Any) -> Any:
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
        source_label = f"{crew_path}: tasks[{index}]"
        task_class = _task_class_from_definition(task_defn, f"{source_label}: type")
        task_kwargs = _task_kwargs_from_definition(
            task_defn,
            agents_map=agents_map,
            task_name_map=task_name_map,
            source=source_label,
            project_root=crew_path.parent,
        )
        try:
            task = task_class(**task_kwargs)
        except ValidationError as exc:
            raise JSONProjectError(f"{source_label}: validation failed: {exc}") from exc

        tasks_list.append(task)
        task_name = task_defn.get("name")
        if isinstance(task_name, str) and task_name:
            task_name_map[task_name] = task

    crew_kwargs = _crew_kwargs_from_definition(
        project.definition,
        agents=[agents_map[name] for name in project.agent_names],
        tasks=tasks_list,
        agents_map=agents_map,
        source=crew_path,
    )

    try:
        crew = Crew(**crew_kwargs)
    except ValidationError as exc:
        raise JSONProjectError(f"{crew_path}: validation failed: {exc}") from exc
    except JSONProjectValidationError:
        raise
    except Exception as exc:
        raise JSONProjectError(f"{crew_path}: failed to load crew: {exc}") from exc

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

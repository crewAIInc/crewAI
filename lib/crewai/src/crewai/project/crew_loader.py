"""Load crew definitions from JSON/JSONC files and produce Crew instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crewai.project.json_loader import (
    JSONProjectError,
    JSONProjectValidationError,
    _crew_kwargs_from_definition,
    _expect_object,
    _find_agent_file,
    _task_kwargs_from_definition,
    load_agent,
    load_jsonc_file,
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
    defn = _expect_object(load_jsonc_file(crew_path), crew_path)

    if agents_dir is None:
        agents_dir = crew_path.parent / "agents"

    agent_names = defn.get("agents", [])
    if not isinstance(agent_names, list) or not agent_names:
        raise JSONProjectError(f"{crew_path}: 'agents' must be a non-empty list")

    agents_map: dict[str, Any] = {}
    for name in agent_names:
        if not isinstance(name, str) or not name:
            raise JSONProjectError(
                f"{crew_path}: each agent reference must be a non-empty string"
            )
        agent_file = _find_agent_file(Path(agents_dir), name)
        if agent_file is None:
            raise FileNotFoundError(
                f"Agent definition for '{name}' not found in {agents_dir} "
                f"(tried {name}.jsonc and {name}.json)"
            )
        agents_map[name] = load_agent(agent_file)

    task_defs = defn.get("tasks", [])
    if not isinstance(task_defs, list) or not task_defs:
        raise JSONProjectError(f"{crew_path}: 'tasks' must be a non-empty list")

    tasks_list: list[Task] = []
    task_name_map: dict[str, Task] = {}

    for index, task_defn in enumerate(task_defs):
        if not isinstance(task_defn, dict):
            raise JSONProjectError(f"{crew_path}: tasks[{index}] must be an object")
        source_label = f"{crew_path}: tasks[{index}]"
        task_kwargs = _task_kwargs_from_definition(
            task_defn,
            agents_map=agents_map,
            task_name_map=task_name_map,
            source=source_label,
        )
        try:
            task = Task(**task_kwargs)
        except ValidationError as exc:
            raise JSONProjectError(
                f"{source_label}: validation failed: {exc}"
            ) from exc

        tasks_list.append(task)
        task_name = task_defn.get("name")
        if isinstance(task_name, str) and task_name:
            task_name_map[task_name] = task

    crew_kwargs = _crew_kwargs_from_definition(
        defn,
        agents=list(agents_map.values()),
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

    return crew, defn.get("inputs", {})


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

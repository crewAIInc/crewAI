"""Load crew definitions from JSON/JSONC files and produce Crew instances."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crewai.project.json_loader import (
    _resolve_tools,
    load_agent,
    strip_jsonc_comments,
)

logger = logging.getLogger(__name__)


def load_crew(
    source: Path | str,
    agents_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a ``Crew`` from a JSON/JSONC definition file.

    The definition file describes the crew's agents (by name reference),
    tasks, process type, and default inputs.  Agent definitions are resolved
    from individual ``<name>.jsonc`` / ``<name>.json`` files inside an
    ``agents/`` directory.

    Args:
        source: Path to a ``.json`` or ``.jsonc`` crew definition file.
        agents_dir: Directory containing per-agent definition files.  When
            ``None``, defaults to ``agents/`` relative to the crew file's
            parent directory.

    Returns:
        A ``(crew, default_inputs)`` tuple where *crew* is a fully
        constructed ``Crew`` instance and *default_inputs* is the ``inputs``
        dict from the definition (may be empty).

    Raises:
        FileNotFoundError: If the crew file or a referenced agent file cannot
            be found.
        KeyError: If a task references an agent name that is not declared in
            the ``agents`` list.
        json.JSONDecodeError: If the crew file contains invalid JSON after
            comment stripping.
    """
    from crewai import Agent, Crew, Process, Task

    # ------------------------------------------------------------------
    # 1. Read & parse the crew definition file
    # ------------------------------------------------------------------
    crew_path = Path(source)
    raw = crew_path.read_text(encoding="utf-8")
    clean = strip_jsonc_comments(raw)
    defn: dict[str, Any] = json.loads(clean)

    # ------------------------------------------------------------------
    # 2. Determine agents directory
    # ------------------------------------------------------------------
    if agents_dir is None:
        agents_dir = crew_path.parent / "agents"

    # ------------------------------------------------------------------
    # 3. Load each agent from its own definition file
    # ------------------------------------------------------------------
    agent_names: list[str] = defn.get("agents", [])
    agents_map: dict[str, Agent] = {}

    for name in agent_names:
        agent_file: Path | None = None
        for ext in (".jsonc", ".json"):
            candidate = agents_dir / f"{name}{ext}"
            if candidate.exists():
                agent_file = candidate
                break

        if agent_file is None:
            raise FileNotFoundError(
                f"Agent definition for '{name}' not found in {agents_dir} "
                f"(tried {name}.jsonc and {name}.json)"
            )

        agents_map[name] = load_agent(agent_file)

    # ------------------------------------------------------------------
    # 4. Build Task instances
    # ------------------------------------------------------------------
    tasks_list: list[Task] = []
    task_name_map: dict[str, Task] = {}

    for task_defn in defn.get("tasks", []):
        task_kwargs: dict[str, Any] = {
            "description": task_defn["description"],
            "expected_output": task_defn["expected_output"],
        }

        # Resolve agent reference
        agent_ref = task_defn.get("agent")
        if agent_ref is not None:
            if agent_ref not in agents_map:
                raise KeyError(
                    f"Task '{task_defn.get('name', '?')}' references agent "
                    f"'{agent_ref}' which is not in the crew's agents list"
                )
            task_kwargs["agent"] = agents_map[agent_ref]

        # Resolve context (list of task name strings -> Task references)
        context_names = task_defn.get("context")
        if context_names:
            context_tasks: list[Task] = []
            for ctx_name in context_names:
                if ctx_name not in task_name_map:
                    raise KeyError(
                        f"Task '{task_defn.get('name', '?')}' has context "
                        f"reference '{ctx_name}' but that task has not been "
                        f"defined yet (tasks must be ordered so that context "
                        f"dependencies come first)"
                    )
                context_tasks.append(task_name_map[ctx_name])
            task_kwargs["context"] = context_tasks

        # Resolve tools
        tool_names = task_defn.get("tools")
        if tool_names:
            task_kwargs["tools"] = _resolve_tools(tool_names)

        # Simple pass-through fields
        if "output_file" in task_defn:
            task_kwargs["output_file"] = task_defn["output_file"]
        if "human_input" in task_defn:
            task_kwargs["human_input"] = bool(task_defn["human_input"])
        if "async_execution" in task_defn:
            task_kwargs["async_execution"] = bool(task_defn["async_execution"])

        task = Task(**task_kwargs)
        tasks_list.append(task)

        # Register in the name map so later tasks can reference it as context
        task_name = task_defn.get("name")
        if task_name:
            task_name_map[task_name] = task

    # ------------------------------------------------------------------
    # 5. Resolve Process enum
    # ------------------------------------------------------------------
    process_str = defn.get("process", "sequential")
    process_map = {
        "sequential": Process.sequential,
        "hierarchical": Process.hierarchical,
    }
    process = process_map.get(process_str, Process.sequential)

    # ------------------------------------------------------------------
    # 6. Construct the Crew
    # ------------------------------------------------------------------
    crew = Crew(
        name=defn.get("name"),
        agents=list(agents_map.values()),
        tasks=tasks_list,
        process=process,
        verbose=defn.get("verbose", False),
        memory=defn.get("memory", False),
        planning=defn.get("planning", False),
        manager_llm=defn.get("manager_llm"),
    )

    # ------------------------------------------------------------------
    # 7. Return crew + default inputs
    # ------------------------------------------------------------------
    return crew, defn.get("inputs", {})


def load_crew_and_kickoff(
    crew_path: Path | str,
    input_overrides: dict[str, Any] | None = None,
) -> Any:
    """Convenience function: load a crew and immediately kick it off.

    Args:
        crew_path: Path to the crew JSON/JSONC definition file.
        input_overrides: Optional dict of input values that override (or
            extend) the defaults declared in the crew file.

    Returns:
        The result of ``crew.kickoff(inputs=...)``.
    """
    crew, default_inputs = load_crew(crew_path)

    merged_inputs = {**default_inputs}
    if input_overrides:
        merged_inputs.update(input_overrides)

    return crew.kickoff(inputs=merged_inputs)

"""Dependency graph resolution for event handlers.

This module resolves handler dependencies into execution levels, ensuring
handlers execute in correct order while maximizing parallelism.
"""

from collections import defaultdict, deque
from collections.abc import Sequence

from crewai.events.depends import Depends
from crewai.events.types.event_bus_types import ExecutionPlan, Handler


class CircularDependencyError(Exception):
    """Exception raised when circular dependencies are detected in event handlers.

    Attributes:
        handlers: The handlers involved in the circular dependency
    """

    def __init__(self, handlers: list[Handler]) -> None:
        """Initialize the circular dependency error.

        Args:
            handlers: The handlers involved in the circular dependency
        """
        handler_names = ", ".join(getattr(h, "__name__", repr(h)) for h in handlers[:5])
        message = f"Circular dependency detected in event handlers: {handler_names}"
        super().__init__(message)
        self.handlers = handlers


class HandlerGraph:
    """Resolves handler dependencies into parallel execution levels.

    Handlers are organized into levels where:
    - Level 0: Handlers with no dependencies (can run first)
    - Level N: Handlers that depend on handlers in levels 0...N-1

    Handlers within the same level can execute in parallel.

    Attributes:
        levels: List of handler sets, where each level can execute in parallel
    """

    def __init__(
        self,
        handlers: dict[Handler, list[Depends]],
    ) -> None:
        """Initialize the dependency graph.

        Args:
            handlers: Mapping of handler -> list of `crewai.events.depends.Depends` objects
        """
        self.handlers = handlers
        self.levels: ExecutionPlan = []
        self._resolve()

    def _resolve(self) -> None:
        """Resolve dependencies into execution levels using topological sort."""
        dependents: dict[Handler, set[Handler]] = defaultdict(set)
        in_degree: dict[Handler, int] = {}

        for handler in self.handlers:
            in_degree[handler] = 0

        for handler, deps in self.handlers.items():
            in_degree[handler] = len(deps)
            for dep in deps:
                dependents[dep.handler].add(handler)

        queue: deque[Handler] = deque([h for h, deg in in_degree.items() if deg == 0])

        while queue:
            current_level: set[Handler] = set()

            for _ in range(len(queue)):
                handler = queue.popleft()
                current_level.add(handler)

                for dependent in dependents[handler]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            if current_level:
                self.levels.append(current_level)

        remaining = [h for h, deg in in_degree.items() if deg > 0]
        if remaining:
            raise CircularDependencyError(remaining)

    def get_execution_plan(self) -> ExecutionPlan:
        """Get the ordered execution plan.

        Returns:
            List of handler sets, where each set represents handlers that can
            execute in parallel. Sets are ordered such that dependencies are
            satisfied.
        """
        return self.levels


def build_execution_plan(
    handlers: Sequence[Handler],
    dependencies: dict[Handler, list[Depends]],
) -> ExecutionPlan:
    """Build an execution plan from handlers and their dependencies.

    Args:
        handlers: All handlers for an event type
        dependencies: Mapping of handler -> list of dependencies

    Returns:
        Execution plan as list of levels, where each level is a set of
        handlers that can execute in parallel

    Raises:
        CircularDependencyError: If circular dependencies are detected
    """
    handler_dict: dict[Handler, list[Depends]] = {
        h: dependencies.get(h, []) for h in handlers
    }

    graph = HandlerGraph(handler_dict)
    return graph.get_execution_plan()

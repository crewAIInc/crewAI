"""Decorators for defining crew components and their behaviors."""

from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

from crewai import Crew
from crewai.project.utils import memoize

P = ParamSpec("P")
R = TypeVar("R")


def before_kickoff(func: Callable[P, R]) -> Callable[P, R]:
    """Marks a method to execute before crew kickoff."""
    func.is_before_kickoff = True  # type: ignore
    return func


def after_kickoff(func: Callable[P, R]) -> Callable[P, R]:
    """Marks a method to execute after crew kickoff."""
    func.is_after_kickoff = True  # type: ignore
    return func


def task(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as a crew task."""
    func.is_task = True  # type: ignore

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        result = func(self, *args, **kwargs)
        if not result.name:  # type: ignore
            result.name = func.__name__  # type: ignore
        return result

    return memoize(wrapper)


def agent(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as a crew agent."""
    func.is_agent = True  # type: ignore
    return memoize(func)


def llm(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as an LLM provider."""
    func.is_llm = True  # type: ignore
    return memoize(func)


def output_json(cls: type[R]) -> type[R]:
    """Marks a class as JSON output format."""
    cls.is_output_json = True  # type: ignore
    return cls


def output_pydantic(cls: type[R]) -> type[R]:
    """Marks a class as Pydantic output format."""
    cls.is_output_pydantic = True  # type: ignore
    return cls


def tool(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as a crew tool."""
    func.is_tool = True  # type: ignore
    return memoize(func)


def callback(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as a crew callback."""
    func.is_callback = True  # type: ignore
    return memoize(func)


def cache_handler(
    func: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    """Marks a method as a cache handler."""
    func.is_cache_handler = True  # type: ignore
    return memoize(func)


def crew(
    func: Callable[Concatenate[Any, P], Crew],
) -> Callable[Concatenate[Any, P], Crew]:
    """Marks a method as the main crew execution point."""

    @wraps(func)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> Crew:
        instantiated_tasks = []
        instantiated_agents = []
        agent_roles = set()

        # Use the preserved task and agent information
        tasks = self._original_tasks.items()
        agents = self._original_agents.items()

        # Instantiate tasks in order
        for _task_name, task_method in tasks:
            task_instance = task_method(self)
            instantiated_tasks.append(task_instance)
            agent_instance = getattr(task_instance, "agent", None)
            if agent_instance and agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        # Instantiate agents not included by tasks
        for _agent_name, agent_method in agents:
            agent_instance = agent_method(self)
            if agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        crew_result = func(self, *args, **kwargs)

        def callback_wrapper(callback_func: Any, instance: Any) -> Callable[..., Any]:
            def inner_wrapper(*cb_args: Any, **cb_kwargs: Any) -> Any:
                return callback_func(instance, *cb_args, **cb_kwargs)

            return inner_wrapper

        for callback_func in self._before_kickoff.values():
            crew_result.before_kickoff_callbacks.append(
                callback_wrapper(callback_func, self)
            )
        for callback_func in self._after_kickoff.values():
            crew_result.after_kickoff_callbacks.append(
                callback_wrapper(callback_func, self)
            )

        return crew_result

    return memoize(wrapper)

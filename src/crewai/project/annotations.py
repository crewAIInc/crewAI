"""Decorators for defining crew components and their behaviors."""

from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

from crewai import Crew
from crewai.project.utils import memoize

P = ParamSpec("P")
R = TypeVar("R")


def before_kickoff(meth):
    """Marks a method to execute before crew kickoff."""
    meth.is_before_kickoff = True
    return meth


def after_kickoff(meth):
    """Marks a method to execute after crew kickoff."""
    meth.is_after_kickoff = True
    return meth


def task(meth):
    """Marks a method as a crew task."""
    meth.is_task = True

    @wraps(meth)
    def wrapper(*args, **kwargs):
        result = meth(*args, **kwargs)
        if not result.name:
            result.name = meth.__name__
        return result

    return memoize(wrapper)


def agent(meth):
    """Marks a method as a crew agent."""
    meth.is_agent = True
    return memoize(meth)


def llm(meth):
    """Marks a method as an LLM provider."""
    meth.is_llm = True
    return memoize(meth)


def output_json(cls):
    """Marks a class as JSON output format."""
    cls.is_output_json = True
    return cls


def output_pydantic(cls):
    """Marks a class as Pydantic output format."""
    cls.is_output_pydantic = True
    return cls


def tool(meth):
    """Marks a method as a crew tool."""
    meth.is_tool = True
    return memoize(meth)


def callback(meth):
    """Marks a method as a crew callback."""
    meth.is_callback = True
    return memoize(meth)


def cache_handler(meth):
    """Marks a method as a cache handler."""
    meth.is_cache_handler = True
    return memoize(meth)


def crew(meth) -> Callable[..., Crew]:
    """Marks a method as the main crew execution point."""

    @wraps(meth)
    def wrapper(self, *args, **kwargs) -> Crew:
        instantiated_tasks = []
        instantiated_agents = []
        agent_roles = set()

        # Use the preserved task and agent information
        tasks = self._original_tasks.items()
        agents = self._original_agents.items()

        # Instantiate tasks in order
        for _, task_method in tasks:
            task_instance = task_method(self)
            instantiated_tasks.append(task_instance)
            agent_instance = getattr(task_instance, "agent", None)
            if agent_instance and agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        # Instantiate agents not included by tasks
        for _, agent_method in agents:
            agent_instance = agent_method(self)
            if agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        crew_instance = meth(self, *args, **kwargs)

        def callback_wrapper(
            hook: Callable[Concatenate[Any, P], R], instance: Any
        ) -> Callable[P, R]:
            def bound_callback(*cb_args: P.args, **cb_kwargs: P.kwargs) -> R:
                return hook(instance, *cb_args, **cb_kwargs)

            return bound_callback

        for hook_callback in self._before_kickoff.values():
            crew_instance.before_kickoff_callbacks.append(
                callback_wrapper(hook_callback, self)
            )
        for hook_callback in self._after_kickoff.values():
            crew_instance.after_kickoff_callbacks.append(
                callback_wrapper(hook_callback, self)
            )

        return crew_instance

    return memoize(wrapper)

"""Decorators for defining crew components and their behaviors."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from crewai.project.utils import memoize

if TYPE_CHECKING:
    from crewai import Agent, Crew, Task
from crewai.project.wrappers import (
    AfterKickoffMethod,
    AgentMethod,
    BeforeKickoffMethod,
    CacheHandlerMethod,
    CallbackMethod,
    CrewInstance,
    LLMMethod,
    OutputJsonClass,
    OutputPydanticClass,
    TaskMethod,
    TaskResultT,
    ToolMethod,
)

P = ParamSpec("P")
P2 = ParamSpec("P2")
R = TypeVar("R")
R2 = TypeVar("R2")
T = TypeVar("T")


def before_kickoff(meth: Callable[P, R]) -> BeforeKickoffMethod[P, R]:
    """Marks a method to execute before crew kickoff.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked for before kickoff execution.
    """
    return BeforeKickoffMethod(meth)


def after_kickoff(meth: Callable[P, R]) -> AfterKickoffMethod[P, R]:
    """Marks a method to execute after crew kickoff.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked for after kickoff execution.
    """
    return AfterKickoffMethod(meth)


def task(meth: Callable[P, TaskResultT]) -> TaskMethod[P, TaskResultT]:
    """Marks a method as a crew task.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as a task with memoization.
    """
    return TaskMethod(memoize(meth))


def agent(meth: Callable[P, R]) -> AgentMethod[P, R]:
    """Marks a method as a crew agent.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as an agent with memoization.
    """
    return AgentMethod(memoize(meth))


def llm(meth: Callable[P, R]) -> LLMMethod[P, R]:
    """Marks a method as an LLM provider.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as an LLM provider with memoization.
    """
    return LLMMethod(memoize(meth))


def output_json(cls: type[T]) -> OutputJsonClass[T]:
    """Marks a class as JSON output format.

    Args:
        cls: The class to mark.

    Returns:
        A wrapped class marked as JSON output format.
    """
    return OutputJsonClass(cls)


def output_pydantic(cls: type[T]) -> OutputPydanticClass[T]:
    """Marks a class as Pydantic output format.

    Args:
        cls: The class to mark.

    Returns:
        A wrapped class marked as Pydantic output format.
    """
    return OutputPydanticClass(cls)


def tool(meth: Callable[P, R]) -> ToolMethod[P, R]:
    """Marks a method as a crew tool.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as a tool with memoization.
    """
    return ToolMethod(memoize(meth))


def callback(meth: Callable[P, R]) -> CallbackMethod[P, R]:
    """Marks a method as a crew callback.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as a callback with memoization.
    """
    return CallbackMethod(memoize(meth))


def cache_handler(meth: Callable[P, R]) -> CacheHandlerMethod[P, R]:
    """Marks a method as a cache handler.

    Args:
        meth: The method to mark.

    Returns:
        A wrapped method marked as a cache handler with memoization.
    """
    return CacheHandlerMethod(memoize(meth))


def crew(
    meth: Callable[Concatenate[CrewInstance, P], Crew],
) -> Callable[Concatenate[CrewInstance, P], Crew]:
    """Marks a method as the main crew execution point.

    Args:
        meth: The method to mark as crew execution point.

    Returns:
        A wrapped method that instantiates tasks and agents before execution.
    """

    @wraps(meth)
    def wrapper(self: CrewInstance, *args: P.args, **kwargs: P.kwargs) -> Crew:
        """Wrapper that sets up crew before calling the decorated method.

        Args:
            self: The crew class instance.
            *args: Additional positional arguments.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The configured Crew instance with callbacks attached.
        """
        instantiated_tasks: list[Task] = []
        instantiated_agents: list[Agent] = []
        agent_roles: set[str] = set()

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
            hook: Callable[Concatenate[CrewInstance, P2], R2], instance: CrewInstance
        ) -> Callable[P2, R2]:
            """Bind a hook callback to an instance.

            Args:
                hook: The callback hook to bind.
                instance: The instance to bind to.

            Returns:
                A bound callback function.
            """

            def bound_callback(*cb_args: P2.args, **cb_kwargs: P2.kwargs) -> R2:
                """Execute the bound callback.

                Args:
                    *cb_args: Positional arguments for the callback.
                    **cb_kwargs: Keyword arguments for the callback.

                Returns:
                    The result of the callback execution.
                """
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

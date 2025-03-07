from functools import wraps
from typing import Any, Callable, Optional, Union, cast

from crewai import Crew
from crewai.project.utils import memoize
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

"""Decorators for defining crew components and their behaviors."""


def before_kickoff(func):
    """Marks a method to execute before crew kickoff."""
    func.is_before_kickoff = True
    return func


def after_kickoff(func):
    """Marks a method to execute after crew kickoff."""
    func.is_after_kickoff = True
    return func


def task(func):
    """Marks a method as a crew task."""
    setattr(func, "is_task", True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Set the task name if not already set
        if hasattr(result, "name") and not result.name:
            result.name = func.__name__

        # If this is a ConditionalTask, ensure it has a valid condition
        if isinstance(result, ConditionalTask):
            # If the condition is a boolean, wrap it in a function
            if isinstance(result.condition, bool):
                bool_value = result.condition
                result.condition = lambda _: bool_value

            # Get the previous task output if available
            self = args[0] if args else None
            if self and hasattr(self, "_method_outputs"):
                previous_outputs = getattr(self, "_method_outputs", [])
                previous_output = previous_outputs[-1] if previous_outputs else None

                # If there's a previous output and it's a TaskOutput, check if we should execute
                if previous_output and isinstance(previous_output, TaskOutput):
                    if not result.should_execute(previous_output):
                        # Return a skipped task output instead of the task
                        return result.get_skipped_task_output()

        return result

    return memoize(wrapper)


def agent(func):
    """Marks a method as a crew agent."""
    func.is_agent = True
    func = memoize(func)
    return func


def llm(func):
    """Marks a method as an LLM provider."""
    func.is_llm = True
    func = memoize(func)
    return func


def output_json(cls):
    """Marks a class as JSON output format."""
    cls.is_output_json = True
    return cls


def output_pydantic(cls):
    """Marks a class as Pydantic output format."""
    cls.is_output_pydantic = True
    return cls


def tool(func):
    """Marks a method as a crew tool."""
    func.is_tool = True
    return memoize(func)


def callback(func):
    """Marks a method as a crew callback."""
    func.is_callback = True
    return memoize(func)


def cache_handler(func):
    """Marks a method as a cache handler."""
    func.is_cache_handler = True
    return memoize(func)


def crew(func) -> Callable[..., Crew]:
    """Marks a method as the main crew execution point."""

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Crew:
        instantiated_tasks = []
        instantiated_agents = []
        agent_roles = set()

        # Use the preserved task and agent information
        tasks = self._original_tasks.items()
        agents = self._original_agents.items()

        # Instantiate tasks in order
        for task_name, task_method in tasks:
            task_instance = task_method(self)
            instantiated_tasks.append(task_instance)
            agent_instance = getattr(task_instance, "agent", None)
            if agent_instance and agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        # Instantiate agents not included by tasks
        for agent_name, agent_method in agents:
            agent_instance = agent_method(self)
            if agent_instance.role not in agent_roles:
                instantiated_agents.append(agent_instance)
                agent_roles.add(agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        crew = func(self, *args, **kwargs)

        def callback_wrapper(callback, instance):
            def wrapper(*args, **kwargs):
                return callback(instance, *args, **kwargs)

            return wrapper

        for _, callback in self._before_kickoff.items():
            crew.before_kickoff_callbacks.append(callback_wrapper(callback, self))
        for _, callback in self._after_kickoff.items():
            crew.after_kickoff_callbacks.append(callback_wrapper(callback, self))

        return crew

    return memoize(wrapper)

from functools import wraps
from typing import Callable

from crewai import Crew
from crewai.project.utils import memoize


def task(func):
    func.is_task = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not result.name:
            result.name = func.__name__
        return result

    return memoize(wrapper)


def agent(func):
    func.is_agent = True
    func = memoize(func)
    return func


def llm(func):
    func.is_llm = True
    func = memoize(func)
    return func


def output_json(cls):
    cls.is_output_json = True
    return cls


def output_pydantic(cls):
    cls.is_output_pydantic = True
    return cls


def tool(func):
    func.is_tool = True
    return memoize(func)


def callback(func):
    func.is_callback = True
    return memoize(func)


def cache_handler(func):
    func.is_cache_handler = True
    return memoize(func)


def stage(func):
    func.is_stage = True
    return memoize(func)


def router(func):
    func.is_router = True
    return memoize(func)


def pipeline(func):
    func.is_pipeline = True
    return memoize(func)


def crew(func) -> Callable[..., Crew]:
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

        return func(self, *args, **kwargs)

    return wrapper

from functools import wraps

from crewai.project.utils import memoize


def task(func):
    if not hasattr(task, "registration_order"):
        task.registration_order = []

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not result.name:
            result.name = func.__name__
        return result

    setattr(wrapper, "is_task", True)
    task.registration_order.append(func.__name__)

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


def crew(func):
    def wrapper(self, *args, **kwargs):
        instantiated_tasks = []
        instantiated_agents = []

        agent_roles = set()
        all_functions = {
            name: getattr(self, name)
            for name in dir(self)
            if callable(getattr(self, name))
        }
        tasks = {
            name: func
            for name, func in all_functions.items()
            if hasattr(func, "is_task")
        }
        agents = {
            name: func
            for name, func in all_functions.items()
            if hasattr(func, "is_agent")
        }

        # Sort tasks by their registration order
        sorted_task_names = sorted(
            tasks, key=lambda name: task.registration_order.index(name)
        )

        # Instantiate tasks in the order they were defined
        for task_name in sorted_task_names:
            task_instance = tasks[task_name]()
            instantiated_tasks.append(task_instance)
            agent_instance = getattr(task_instance, "agent", None)
            if agent_instance is not None:
                agent_instance = task_instance.agent
                if agent_instance.role not in agent_roles:
                    instantiated_agents.append(agent_instance)
                    agent_roles.add(agent_instance.role)

        # Instantiate any additional agents not already included by tasks
        for agent_name in agents:
            temp_agent_instance = agents[agent_name]()
            if temp_agent_instance.role not in agent_roles:
                instantiated_agents.append(temp_agent_instance)
                agent_roles.add(temp_agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        return func(self, *args, **kwargs)

    return wrapper

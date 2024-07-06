def memoize(func):
    cache = {}

    def memoized_func(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    memoized_func.__dict__.update(func.__dict__)
    return memoized_func


def task(func):
    if not hasattr(task, "registration_order"):
        task.registration_order = []

    func.is_task = True
    wrapped_func = memoize(func)

    # Append the function name to the registration order list
    task.registration_order.append(func.__name__)

    return wrapped_func


def agent(func):
    func.is_agent = True
    func = memoize(func)
    return func


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
            if hasattr(task_instance, "agent"):
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

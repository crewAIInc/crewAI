def task(func):
    func.is_task = True
    return func


def agent(func):
    func.is_agent = True
    return func


def crew(func):
    def wrapper(self, *args, **kwargs):
        instantiated_tasks = []
        instantiated_agents = []

        # Instantiate tasks and collect their associated agents
        agent_roles = set()
        for attr_name in dir(self):
            possible_task = getattr(self, attr_name)
            if callable(possible_task) and hasattr(possible_task, "is_task"):
                task_instance = possible_task()
                instantiated_tasks.append(task_instance)
                if hasattr(task_instance, "agent"):
                    agent_instance = task_instance.agent
                    if agent_instance.role not in agent_roles:
                        instantiated_agents.append(agent_instance)
                        agent_roles.add(agent_instance.role)

        # Instantiate any additional agents not already included by tasks
        for attr_name in dir(self):
            possible_agent = getattr(self, attr_name)
            if callable(possible_agent) and hasattr(possible_agent, "is_agent"):
                temp_agent_instance = possible_agent()
                if temp_agent_instance.role not in agent_roles:
                    instantiated_agents.append(temp_agent_instance)
                    agent_roles.add(temp_agent_instance.role)

        self.agents = instantiated_agents
        self.tasks = instantiated_tasks

        # Now call the original crew method
        return func(self, *args, **kwargs)

    return wrapper

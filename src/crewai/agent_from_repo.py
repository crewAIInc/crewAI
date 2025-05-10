def _load_agent_from_repository(self):
    from crewai.cli.authentication.token import get_auth_token
    from crewai.cli.plus_api import PlusAPI

    if self.from_repository:
        agent = PlusAPI(api_key=get_auth_token()).get_agent(self.from_repository)
        breakpoint()
        if agent:
            self.role = agent["role"]
            self.goal = agent["goal"]
            self.backstory = agent["backstory"]

            import importlib

            for tool_name in agent["tools"]:
                module = importlib.import_module("crewai_tools")
                tool_class = getattr(module, tool_name)
                self.tools.append(tool_class())

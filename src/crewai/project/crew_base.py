import inspect
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ConfigDict

load_dotenv()


def CrewBase(cls):
    class WrappedClass(cls):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        is_crew_class: bool = True  # type: ignore

        base_directory = None
        for frame_info in inspect.stack():
            if "site-packages" not in frame_info.filename:
                base_directory = Path(frame_info.filename).parent.resolve()
                break

        if base_directory is None:
            raise Exception(
                "Unable to dynamically determine the project's base directory, you must run it from the project's root directory."
            )

        original_agents_config_path = getattr(
            cls, "agents_config", "config/agents.yaml"
        )
        original_tasks_config_path = getattr(cls, "tasks_config", "config/tasks.yaml")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.agents_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_agents_config_path)  # type: ignore
            )
            self.tasks_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_tasks_config_path)  # type: ignore
            )
            self.map_all_agent_variables()
            self.map_all_task_varibles()

        @staticmethod
        def load_yaml(config_path: str):
            with open(config_path, "r") as file:
                # parsedContent = YamlParser.parse(file)  # type: ignore # Argument 1 to "parse" has incompatible type "TextIOWrapper"; expected "YamlParser"
                return yaml.safe_load(file)

        def _get_all_functions(self):
            return {
                name: getattr(self, name)
                for name in dir(self)
                if callable(getattr(self, name))
            }

        # def map_string_agent_to_task_agent(self):
        #     all_functions = self._get_all_functions()
        #     agents = {
        #         name: func
        #         for name, func in all_functions.items()
        #         if hasattr(func, "is_agent")
        #     }

        #     for task_name, task_info in self.tasks_config.items():
        #         agent_name = task_info.get("agent")

        #         if agent_name:
        #             self.tasks_config[task_name]["agent"] = agents[agent_name]()

        # def map_task_contexts(self):
        #     all_functions = self._get_all_functions()
        #     tasks = {
        #         name: func
        #         for name, func in all_functions.items()
        #         if hasattr(func, "is_task")
        #     }
        #     for task_name, task_info in self.tasks_config.items():
        #         context_converted = []
        #         context_list = task_info.get("context")

        #         if context_list:
        #             for context_task_name in context_list:
        #                 context_converted.append(tasks[context_task_name]())

        #             self.tasks_config[task_name]["context"] = context_converted

        # def map_output_jsons(self):
        #     all_functions = self._get_all_functions()

        #     output_json_functions = {
        #         name: cls
        #         for name, cls in all_functions.items()
        #         if hasattr(cls, "output_json")
        #     }

        #     for task_name, task_info in self.tasks_config.items():
        #         output_json = task_info.get("output_json")
        #         if output_json:
        #             self.tasks_config[task_name]["output_json"] = output_json_functions[
        #                 output_json
        #             ]

        def map_all_agent_variables(self):
            all_functions = self._get_all_functions()
            agents = {
                name: func
                for name, func in all_functions.items()
                if hasattr(func, "is_agent")
            }
            for agent_name, agent_info in self.agents_config.items():
                tools = agent_info.get("tools")
                function_calling_llm = agent_info.get("function_calling_llm")
                step_callback = agent_info.get("step_callback")

                if tools:
                    self.agents_config[agent_name]["tools"] = [
                        agents[tool]() for tool in tools
                    ]
                if function_calling_llm:
                    self.agents_config[agent_name]["function_calling_llm"] = agents[
                        function_calling_llm
                    ]()
                if step_callback:
                    self.agents_config[agent_name]["step_callback"] = agents[
                        step_callback
                    ]()

        def map_all_task_varibles(self):
            all_functions = self._get_all_functions()
            agents = {
                name: func
                for name, func in all_functions.items()
                if hasattr(func, "is_agent")
            }

            tasks = {
                name: func
                for name, func in all_functions.items()
                if hasattr(func, "is_task")
            }
            output_json_functions = {
                name: cls
                for name, cls in all_functions.items()
                if hasattr(cls, "is_output_json")
            }

            tool_functions = {
                name: func
                for name, func in all_functions.items()
                if hasattr(func, "is_tool")
            }
            callback_functions = {
                name: func
                for name, func in all_functions.items()
                if hasattr(func, "is_callback")
            }
            output_pydantic_functions = {
                name: cls
                for name, cls in all_functions.items()
                if hasattr(cls, "is_output_pydantic")
            }

            for task_name, task_info in self.tasks_config.items():
                context_converted = []
                context_list = task_info.get("context")
                agent_name = task_info.get("agent")
                output_json = task_info.get("output_json")
                output_pydantic = task_info.get("output_pydantic")
                tools = task_info.get("tools")
                callbacks = task_info.get("callbacks")

                if context_list:
                    for context_task_name in context_list:
                        context_converted.append(tasks[context_task_name]())

                    self.tasks_config[task_name]["context"] = context_converted

                if tools:
                    self.tasks_config[task_name]["tools"] = [
                        tool_functions[tool]() for tool in tools
                    ]

                if agent_name:
                    self.tasks_config[task_name]["agent"] = agents[agent_name]()

                if output_json:
                    self.tasks_config[task_name]["output_json"] = output_json_functions[
                        output_json
                    ]

                if output_pydantic:
                    self.tasks_config[task_name]["output_pydantic"] = (
                        output_pydantic_functions[output_pydantic]
                    )

                if callbacks:
                    self.tasks_config[task_name]["callbacks"] = [
                        callback_functions[callback]() for callback in callbacks
                    ]

        # def map_output_pydantics(self):
        #     all_functions = self._get_all_functions()

        #     output_pydantic_functions = {
        #         name: cls
        #         for name, cls in all_functions.items()
        #         if hasattr(cls, "output_pydantic")
        #     }

        #     for task_name, task_info in self.tasks_config.items():
        #         output_pydantic = task_info.get("output_pydantic")
        #         if output_pydantic:
        #             self.tasks_config[task_name]["output_pydantic"] = (
        #                 output_pydantic_functions[output_pydantic]
        #             )

        # def _map_configs(self):
        #     self.map_string_agent_to_task_agent()
        #     self.map_task_contexts()
        #     self.map_output_jsons()
        #     self.map_output_pydantics()

    return WrappedClass

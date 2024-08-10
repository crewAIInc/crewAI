import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict

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

        original_agents_config_path = getattr(
            cls, "agents_config", "config/agents.yaml"
        )

        original_tasks_config_path = getattr(cls, "tasks_config", "config/tasks.yaml")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if self.base_directory is None:
                raise Exception(
                    "Unable to dynamically determine the project's base directory, you must run it from the project's root directory."
                )

            self.agents_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_agents_config_path)
            )

            self.tasks_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_tasks_config_path)
            )

            self.map_all_agent_variables()
            self.map_all_task_variables()

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

        def _filter_functions(
            self, functions: Dict[str, Callable], attribute: str
        ) -> Dict[str, Callable]:
            return {
                name: func
                for name, func in functions.items()
                if hasattr(func, attribute)
            }

        def map_all_agent_variables(self) -> None:
            all_functions = self._get_all_functions()
            llms = self._filter_functions(all_functions, "is_llm")
            tool_functions = self._filter_functions(all_functions, "is_tool")
            cache_handler_functions = self._filter_functions(
                all_functions, "is_cache_handler"
            )
            callbacks = self._filter_functions(all_functions, "is_callback")
            agents = self._filter_functions(all_functions, "is_agent")

            for agent_name, agent_info in self.agents_config.items():
                self._map_agent_variables(
                    agent_name,
                    agent_info,
                    agents,
                    llms,
                    tool_functions,
                    cache_handler_functions,
                    callbacks,
                )

        def _map_agent_variables(
            self,
            agent_name: str,
            agent_info: Dict[str, Any],
            agents: Dict[str, Callable],
            llms: Dict[str, Callable],
            tool_functions: Dict[str, Callable],
            cache_handler_functions: Dict[str, Callable],
            callbacks: Dict[str, Callable],
        ) -> None:
            if llm := agent_info.get("llm"):
                self.agents_config[agent_name]["llm"] = llms[llm]()

            if tools := agent_info.get("tools"):
                self.agents_config[agent_name]["tools"] = [
                    tool_functions[tool]() for tool in tools
                ]

            if function_calling_llm := agent_info.get("function_calling_llm"):
                self.agents_config[agent_name]["function_calling_llm"] = agents[
                    function_calling_llm
                ]()

            if step_callback := agent_info.get("step_callback"):
                self.agents_config[agent_name]["step_callback"] = callbacks[
                    step_callback
                ]()

            if cache_handler := agent_info.get("cache_handler"):
                self.agents_config[agent_name]["cache_handler"] = (
                    cache_handler_functions[cache_handler]()
                )

        def map_all_task_variables(self) -> None:
            all_functions = self._get_all_functions()
            agents = self._filter_functions(all_functions, "is_agent")
            tasks = self._filter_functions(all_functions, "is_task")
            output_json_functions = self._filter_functions(
                all_functions, "is_output_json"
            )
            tool_functions = self._filter_functions(all_functions, "is_tool")
            callback_functions = self._filter_functions(all_functions, "is_callback")
            output_pydantic_functions = self._filter_functions(
                all_functions, "is_output_pydantic"
            )

            for task_name, task_info in self.tasks_config.items():
                self._map_task_variables(
                    task_name,
                    task_info,
                    agents,
                    tasks,
                    output_json_functions,
                    tool_functions,
                    callback_functions,
                    output_pydantic_functions,
                )

        def _map_task_variables(
            self,
            task_name: str,
            task_info: Dict[str, Any],
            agents: Dict[str, Callable],
            tasks: Dict[str, Callable],
            output_json_functions: Dict[str, Callable],
            tool_functions: Dict[str, Callable],
            callback_functions: Dict[str, Callable],
            output_pydantic_functions: Dict[str, Callable],
        ) -> None:
            if context_list := task_info.get("context"):
                self.tasks_config[task_name]["context"] = [
                    tasks[context_task_name]() for context_task_name in context_list
                ]

            if tools := task_info.get("tools"):
                self.tasks_config[task_name]["tools"] = [
                    tool_functions[tool]() for tool in tools
                ]

            if agent_name := task_info.get("agent"):
                self.tasks_config[task_name]["agent"] = agents[agent_name]()

            if output_json := task_info.get("output_json"):
                self.tasks_config[task_name]["output_json"] = output_json_functions[
                    output_json
                ]

            if output_pydantic := task_info.get("output_pydantic"):
                self.tasks_config[task_name]["output_pydantic"] = (
                    output_pydantic_functions[output_pydantic]
                )

            if callbacks := task_info.get("callbacks"):
                self.tasks_config[task_name]["callbacks"] = [
                    callback_functions[callback]() for callback in callbacks
                ]

    return WrappedClass

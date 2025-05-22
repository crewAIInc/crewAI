import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, cast

import yaml
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=type)

"""Base decorator for creating crew classes with configuration and function management."""


def CrewBase(cls: T) -> T:
    """Wraps a class with crew functionality and configuration management."""

    class WrappedClass(cls):  # type: ignore
        is_crew_class: bool = True  # type: ignore

        # Get the directory of the class being decorated
        base_directory = Path(inspect.getfile(cls)).parent

        original_agents_config_path = getattr(
            cls, "agents_config", "config/agents.yaml"
        )
        original_tasks_config_path = getattr(cls, "tasks_config", "config/tasks.yaml")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load_configurations()
            self.map_all_agent_variables()
            self.map_all_task_variables()
            # Preserve all decorated functions
            self._original_functions = {
                name: method
                for name, method in cls.__dict__.items()
                if any(
                    hasattr(method, attr)
                    for attr in [
                        "is_task",
                        "is_agent",
                        "is_before_kickoff",
                        "is_after_kickoff",
                        "is_kickoff",
                    ]
                )
            }
            # Store specific function types
            self._original_tasks = self._filter_functions(
                self._original_functions, "is_task"
            )
            self._original_agents = self._filter_functions(
                self._original_functions, "is_agent"
            )
            self._before_kickoff = self._filter_functions(
                self._original_functions, "is_before_kickoff"
            )
            self._after_kickoff = self._filter_functions(
                self._original_functions, "is_after_kickoff"
            )
            self._kickoff = self._filter_functions(
                self._original_functions, "is_kickoff"
            )

        def load_configurations(self):
            """Load agent and task configurations from YAML files."""
            if isinstance(self.original_agents_config_path, str):
                agents_config_path = (
                    self.base_directory / self.original_agents_config_path
                )
                try:
                    self.agents_config = self.load_yaml(agents_config_path)
                except FileNotFoundError:
                    logging.warning(
                        f"Agent config file not found at {agents_config_path}. "
                        "Proceeding with empty agent configurations."
                    )
                    self.agents_config = {}
            else:
                logging.warning(
                    "No agent configuration path provided. Proceeding with empty agent configurations."
                )
                self.agents_config = {}

            if isinstance(self.original_tasks_config_path, str):
                tasks_config_path = (
                    self.base_directory / self.original_tasks_config_path
                )
                try:
                    self.tasks_config = self.load_yaml(tasks_config_path)
                except FileNotFoundError:
                    logging.warning(
                        f"Task config file not found at {tasks_config_path}. "
                        "Proceeding with empty task configurations."
                    )
                    self.tasks_config = {}
            else:
                logging.warning(
                    "No task configuration path provided. Proceeding with empty task configurations."
                )
                self.tasks_config = {}

        @staticmethod
        def load_yaml(config_path: Path):
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    return yaml.safe_load(file)
            except FileNotFoundError:
                print(f"File not found: {config_path}")
                raise

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

            for agent_name, agent_info in self.agents_config.items():
                self._map_agent_variables(
                    agent_name,
                    agent_info,
                    llms,
                    tool_functions,
                    cache_handler_functions,
                    callbacks,
                )

        def _map_agent_variables(
            self,
            agent_name: str,
            agent_info: Dict[str, Any],
            llms: Dict[str, Callable],
            tool_functions: Dict[str, Callable],
            cache_handler_functions: Dict[str, Callable],
            callbacks: Dict[str, Callable],
        ) -> None:
            if llm := agent_info.get("llm"):
                try:
                    self.agents_config[agent_name]["llm"] = llms[llm]()
                except KeyError:
                    self.agents_config[agent_name]["llm"] = llm

            if tools := agent_info.get("tools"):
                self.agents_config[agent_name]["tools"] = [
                    tool_functions[tool]() for tool in tools
                ]

            if function_calling_llm := agent_info.get("function_calling_llm"):
                try:
                    self.agents_config[agent_name]["function_calling_llm"] = llms[function_calling_llm]()
                except KeyError:
                    self.agents_config[agent_name]["function_calling_llm"] = function_calling_llm

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

            if guardrail := task_info.get("guardrail"):
                self.tasks_config[task_name]["guardrail"] = guardrail

    # Include base class (qual)name in the wrapper class (qual)name.
    WrappedClass.__name__ = CrewBase.__name__ + "(" + cls.__name__ + ")"
    WrappedClass.__qualname__ = CrewBase.__qualname__ + "(" + cls.__name__ + ")"
    WrappedClass._crew_name = cls.__name__

    return cast(T, WrappedClass)

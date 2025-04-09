import ast
import datetime
import json
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from json import JSONDecodeError
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import json5
from json_repair import repair_json

from crewai.agents.tools_handler import ToolsHandler
from crewai.task import Task
from crewai.telemetry import Telemetry
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.utilities import I18N, Converter, Printer
from crewai.utilities.agent_utils import (
    get_tool_names,
    render_text_description_and_args,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.tool_usage_events import (
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolValidateInputErrorEvent,
)

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.lite_agent import LiteAgent

OPENAI_BIGGER_MODELS = [
    "gpt-4",
    "gpt-4o",
    "o1-preview",
    "o1-mini",
    "o1",
    "o3",
    "o3-mini",
]


class ToolUsageErrorException(Exception):
    """Exception raised for errors in the tool usage."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ToolUsage:
    """
    Class that represents the usage of a tool by an agent.

    Attributes:
      task: Task being executed.
      tools_handler: Tools handler that will manage the tool usage.
      tools: List of tools available for the agent.
      original_tools: Original tools available for the agent before being converted to BaseTool.
      tools_description: Description of the tools available for the agent.
      tools_names: Names of the tools available for the agent.
      function_calling_llm: Language model to be used for the tool usage.
    """

    def __init__(
        self,
        tools_handler: Optional[ToolsHandler],
        tools: List[CrewStructuredTool],
        task: Optional[Task],
        function_calling_llm: Any,
        agent: Optional[Union["BaseAgent", "LiteAgent"]] = None,
        action: Any = None,
        fingerprint_context: Optional[Dict[str, str]] = None,
    ) -> None:
        self._i18n: I18N = agent.i18n if agent else I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remember_format_after_usages: int = 3
        self.agent = agent
        self.tools_description = render_text_description_and_args(tools)
        self.tools_names = get_tool_names(tools)
        self.tools_handler = tools_handler
        self.tools = tools
        self.task = task
        self.action = action
        self.function_calling_llm = function_calling_llm
        self.fingerprint_context = fingerprint_context or {}

        # Set the maximum parsing attempts for bigger models
        if (
            self.function_calling_llm
            and self.function_calling_llm in OPENAI_BIGGER_MODELS
        ):
            self._max_parsing_attempts = 2
            self._remember_format_after_usages = 4

    def parse_tool_calling(self, tool_string: str):
        """Parse the tool string and return the tool calling."""
        return self._tool_calling(tool_string)

    def use(
        self, calling: Union[ToolCalling, InstructorToolCalling], tool_string: str
    ) -> str:
        if isinstance(calling, ToolUsageErrorException):
            error = calling.message
            if self.agent and self.agent.verbose:
                self._printer.print(content=f"\n\n{error}\n", color="red")
            if self.task:
                self.task.increment_tools_errors()
            return error

        try:
            tool = self._select_tool(calling.tool_name)
        except Exception as e:
            error = getattr(e, "message", str(e))
            if self.task:
                self.task.increment_tools_errors()
            if self.agent and self.agent.verbose:
                self._printer.print(content=f"\n\n{error}\n", color="red")
            return error

        if (
            isinstance(tool, CrewStructuredTool)
            and tool.name == self._i18n.tools("add_image")["name"]  # type: ignore
        ):
            try:
                result = self._use(tool_string=tool_string, tool=tool, calling=calling)
                return result

            except Exception as e:
                error = getattr(e, "message", str(e))
                if self.task:
                    self.task.increment_tools_errors()
                if self.agent and self.agent.verbose:
                    self._printer.print(content=f"\n\n{error}\n", color="red")
                return error

        return f"{self._use(tool_string=tool_string, tool=tool, calling=calling)}"

    def _use(
        self,
        tool_string: str,
        tool: CrewStructuredTool,
        calling: Union[ToolCalling, InstructorToolCalling],
    ) -> str:
        if self._check_tool_repeated_usage(calling=calling):  # type: ignore # _check_tool_repeated_usage of "ToolUsage" does not return a value (it only ever returns None)
            try:
                result = self._i18n.errors("task_repeated_usage").format(
                    tool_names=self.tools_names
                )
                self._telemetry.tool_repeated_usage(
                    llm=self.function_calling_llm,
                    tool_name=tool.name,
                    attempts=self._run_attempts,
                )
                result = self._format_result(result=result)  # type: ignore #  "_format_result" of "ToolUsage" does not return a value (it only ever returns None)
                return result  # type: ignore # Fix the return type of this function

            except Exception:
                if self.task:
                    self.task.increment_tools_errors()

        started_at = time.time()
        from_cache = False
        result = None  # type: ignore

        if self.tools_handler and self.tools_handler.cache:
            result = self.tools_handler.cache.read(
                tool=calling.tool_name, input=calling.arguments
            )  # type: ignore
            from_cache = result is not None

        available_tool = next(
            (
                available_tool
                for available_tool in self.tools
                if available_tool.name == tool.name
            ),
            None,
        )

        if result is None:
            try:
                if calling.tool_name in [
                    "Delegate work to coworker",
                    "Ask question to coworker",
                ]:
                    coworker = (
                        calling.arguments.get("coworker") if calling.arguments else None
                    )
                    if self.task:
                        self.task.increment_delegations(coworker)

                if calling.arguments:
                    try:
                        acceptable_args = tool.args_schema.model_json_schema()[
                            "properties"
                        ].keys()  # type: ignore
                        arguments = {
                            k: v
                            for k, v in calling.arguments.items()
                            if k in acceptable_args
                        }
                        # Add fingerprint metadata if available
                        arguments = self._add_fingerprint_metadata(arguments)
                        result = tool.invoke(input=arguments)
                    except Exception:
                        arguments = calling.arguments
                        # Add fingerprint metadata if available
                        arguments = self._add_fingerprint_metadata(arguments)
                        result = tool.invoke(input=arguments)
                else:
                    # Add fingerprint metadata even to empty arguments
                    arguments = self._add_fingerprint_metadata({})
                    result = tool.invoke(input=arguments)
            except Exception as e:
                self.on_tool_error(tool=tool, tool_calling=calling, e=e)
                self._run_attempts += 1
                if self._run_attempts > self._max_parsing_attempts:
                    self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                    error_message = self._i18n.errors("tool_usage_exception").format(
                        error=e, tool=tool.name, tool_inputs=tool.description
                    )
                    error = ToolUsageErrorException(
                        f"\n{error_message}.\nMoving on then. {self._i18n.slice('format').format(tool_names=self.tools_names)}"
                    ).message
                    if self.task:
                        self.task.increment_tools_errors()
                    if self.agent and self.agent.verbose:
                        self._printer.print(
                            content=f"\n\n{error_message}\n", color="red"
                        )
                    return error  # type: ignore # No return value expected

                if self.task:
                    self.task.increment_tools_errors()
                return self.use(calling=calling, tool_string=tool_string)  # type: ignore # No return value expected

            if self.tools_handler:
                should_cache = True
                if (
                    hasattr(available_tool, "cache_function")
                    and available_tool.cache_function  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                ):
                    should_cache = available_tool.cache_function(  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                        calling.arguments, result
                    )

                self.tools_handler.on_tool_use(
                    calling=calling, output=result, should_cache=should_cache
                )
        self._telemetry.tool_usage(
            llm=self.function_calling_llm,
            tool_name=tool.name,
            attempts=self._run_attempts,
        )
        result = self._format_result(result=result)  # type: ignore # "_format_result" of "ToolUsage" does not return a value (it only ever returns None)
        data = {
            "result": result,
            "tool_name": tool.name,
            "tool_args": calling.arguments,
        }

        self.on_tool_use_finished(
            tool=tool,
            tool_calling=calling,
            from_cache=from_cache,
            started_at=started_at,
            result=result,
        )

        if (
            hasattr(available_tool, "result_as_answer")
            and available_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
        ):
            result_as_answer = available_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "result_as_answer"
            data["result_as_answer"] = result_as_answer  # type: ignore

        if self.agent and hasattr(self.agent, "tools_results"):
            self.agent.tools_results.append(data)

        return result

    def _format_result(self, result: Any) -> str:
        if self.task:
            self.task.used_tools += 1
        if self._should_remember_format():
            result = self._remember_format(result=result)
        return str(result)

    def _should_remember_format(self) -> bool:
        if self.task:
            return self.task.used_tools % self._remember_format_after_usages == 0
        return False

    def _remember_format(self, result: str) -> str:
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> bool:
        if not self.tools_handler:
            return False
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (
                calling.arguments == last_tool_usage.arguments
            )
        return False

    def _select_tool(self, tool_name: str) -> Any:
        order_tools = sorted(
            self.tools,
            key=lambda tool: SequenceMatcher(
                None, tool.name.lower().strip(), tool_name.lower().strip()
            ).ratio(),
            reverse=True,
        )
        for tool in order_tools:
            if (
                tool.name.lower().strip() == tool_name.lower().strip()
                or SequenceMatcher(
                    None, tool.name.lower().strip(), tool_name.lower().strip()
                ).ratio()
                > 0.85
            ):
                return tool
        if self.task:
            self.task.increment_tools_errors()
        tool_selection_data: Dict[str, Any] = {
            "agent_key": getattr(self.agent, "key", None) if self.agent else None,
            "agent_role": getattr(self.agent, "role", None) if self.agent else None,
            "tool_name": tool_name,
            "tool_args": {},
            "tool_class": self.tools_description,
        }
        if tool_name and tool_name != "":
            error = f"Action '{tool_name}' don't exist, these are the only available Actions:\n{self.tools_description}"
            crewai_event_bus.emit(
                self,
                ToolSelectionErrorEvent(
                    **tool_selection_data,
                    error=error,
                ),
            )
            raise Exception(error)
        else:
            error = f"I forgot the Action name, these are the only available Actions: {self.tools_description}"
            crewai_event_bus.emit(
                self,
                ToolSelectionErrorEvent(
                    **tool_selection_data,
                    error=error,
                ),
            )
            raise Exception(error)

    def _render(self) -> str:
        """Render the tool name and description in plain text."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.description)
        return "\n--\n".join(descriptions)

    def _function_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling]:
        model = (
            InstructorToolCalling
            if self.function_calling_llm.supports_function_calling()
            else ToolCalling
        )
        converter = Converter(
            text=f"Only tools available:\n###\n{self._render()}\n\nReturn a valid schema for the tool, the tool name must be exactly equal one of the options, use this text to inform the valid output schema:\n\n### TEXT \n{tool_string}",
            llm=self.function_calling_llm,
            model=model,
            instructions=dedent(
                """\
        The schema should have the following structure, only two keys:
        - tool_name: str
        - arguments: dict (always a dictionary, with all arguments being passed)

        Example:
        {"tool_name": "tool name", "arguments": {"arg_name1": "value", "arg_name2": 2}}""",
            ),
            max_attempts=1,
        )
        tool_object = converter.to_pydantic()
        if not isinstance(tool_object, (ToolCalling, InstructorToolCalling)):
            raise ToolUsageErrorException("Failed to parse tool calling")

        return tool_object

    def _original_tool_calling(
        self, tool_string: str, raise_error: bool = False
    ) -> Union[ToolCalling, InstructorToolCalling, ToolUsageErrorException]:
        tool_name = self.action.tool
        tool = self._select_tool(tool_name)
        try:
            arguments = self._validate_tool_input(self.action.tool_input)

        except Exception:
            if raise_error:
                raise
            else:
                return ToolUsageErrorException(
                    f"{self._i18n.errors('tool_arguments_error')}"
                )

        if not isinstance(arguments, dict):
            if raise_error:
                raise
            else:
                return ToolUsageErrorException(
                    f"{self._i18n.errors('tool_arguments_error')}"
                )

        return ToolCalling(
            tool_name=tool.name,
            arguments=arguments,
        )

    def _tool_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling, ToolUsageErrorException]:
        try:
            try:
                return self._original_tool_calling(tool_string, raise_error=True)
            except Exception:
                if self.function_calling_llm:
                    return self._function_calling(tool_string)
                else:
                    return self._original_tool_calling(tool_string)
        except Exception as e:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                if self.task:
                    self.task.increment_tools_errors()
                if self.agent and self.agent.verbose:
                    self._printer.print(content=f"\n\n{e}\n", color="red")
                return ToolUsageErrorException(  # type: ignore # Incompatible return value type (got "ToolUsageErrorException", expected "ToolCalling | InstructorToolCalling")
                    f"{self._i18n.errors('tool_usage_error').format(error=e)}\nMoving on then. {self._i18n.slice('format').format(tool_names=self.tools_names)}"
                )
            return self._tool_calling(tool_string)

    def _validate_tool_input(self, tool_input: Optional[str]) -> Dict[str, Any]:
        if tool_input is None:
            return {}

        if not isinstance(tool_input, str) or not tool_input.strip():
            raise Exception(
                "Tool input must be a valid dictionary in JSON or Python literal format"
            )

        # Attempt 1: Parse as JSON
        try:
            arguments = json.loads(tool_input)
            if isinstance(arguments, dict):
                return arguments
        except (JSONDecodeError, TypeError):
            pass  # Continue to the next parsing attempt

        # Attempt 2: Parse as Python literal
        try:
            arguments = ast.literal_eval(tool_input)
            if isinstance(arguments, dict):
                return arguments
        except (ValueError, SyntaxError):
            repaired_input = repair_json(tool_input)
            pass  # Continue to the next parsing attempt

        # Attempt 3: Parse as JSON5
        try:
            arguments = json5.loads(tool_input)
            if isinstance(arguments, dict):
                return arguments
        except (JSONDecodeError, ValueError, TypeError):
            pass  # Continue to the next parsing attempt

        # Attempt 4: Repair JSON
        try:
            repaired_input = str(repair_json(tool_input, skip_json_loads=True))
            self._printer.print(
                content=f"Repaired JSON: {repaired_input}", color="blue"
            )
            arguments = json.loads(repaired_input)
            if isinstance(arguments, dict):
                return arguments
        except Exception as e:
            error = f"Failed to repair JSON: {e}"
            self._printer.print(content=error, color="red")

        error_message = (
            "Tool input must be a valid dictionary in JSON or Python literal format"
        )
        self._emit_validate_input_error(error_message)
        # If all parsing attempts fail, raise an error
        raise Exception(error_message)

    def _emit_validate_input_error(self, final_error: str):
        tool_selection_data = {
            "agent_key": getattr(self.agent, "key", None) if self.agent else None,
            "agent_role": getattr(self.agent, "role", None) if self.agent else None,
            "tool_name": self.action.tool,
            "tool_args": str(self.action.tool_input),
            "tool_class": self.__class__.__name__,
            "agent": self.agent,  # Adding agent for fingerprint extraction
        }

        # Include fingerprint context if available
        if self.fingerprint_context:
            tool_selection_data.update(self.fingerprint_context)

        crewai_event_bus.emit(
            self,
            ToolValidateInputErrorEvent(**tool_selection_data, error=final_error),
        )

    def on_tool_error(
        self,
        tool: Any,
        tool_calling: Union[ToolCalling, InstructorToolCalling],
        e: Exception,
    ) -> None:
        event_data = self._prepare_event_data(tool, tool_calling)
        crewai_event_bus.emit(self, ToolUsageErrorEvent(**{**event_data, "error": e}))

    def on_tool_use_finished(
        self,
        tool: Any,
        tool_calling: Union[ToolCalling, InstructorToolCalling],
        from_cache: bool,
        started_at: float,
        result: Any,
    ) -> None:
        finished_at = time.time()
        event_data = self._prepare_event_data(tool, tool_calling)
        event_data.update(
            {
                "started_at": datetime.datetime.fromtimestamp(started_at),
                "finished_at": datetime.datetime.fromtimestamp(finished_at),
                "from_cache": from_cache,
                "output": result,
            }
        )
        crewai_event_bus.emit(self, ToolUsageFinishedEvent(**event_data))

    def _prepare_event_data(
        self, tool: Any, tool_calling: Union[ToolCalling, InstructorToolCalling]
    ) -> dict:
        event_data = {
            "run_attempts": self._run_attempts,
            "delegations": self.task.delegations if self.task else 0,
            "tool_name": tool.name,
            "tool_args": tool_calling.arguments,
            "tool_class": tool.__class__.__name__,
            "agent_key": (
                getattr(self.agent, "key", "unknown") if self.agent else "unknown"
            ),
            "agent_role": (
                getattr(self.agent, "_original_role", None)
                or getattr(self.agent, "role", "unknown")
                if self.agent
                else "unknown"
            ),
        }

        # Include fingerprint context if available
        if self.fingerprint_context:
            event_data.update(self.fingerprint_context)

        return event_data

    def _add_fingerprint_metadata(self, arguments: dict) -> dict:
        """Add fingerprint metadata to tool arguments if available.

        Args:
            arguments: The original tool arguments

        Returns:
            Updated arguments dictionary with fingerprint metadata
        """
        # Create a shallow copy to avoid modifying the original
        arguments = arguments.copy()

        # Add security metadata under a designated key
        if "security_context" not in arguments:
            arguments["security_context"] = {}

        security_context = arguments["security_context"]

        # Add agent fingerprint if available
        if self.agent and hasattr(self.agent, "security_config"):
            security_config = getattr(self.agent, "security_config", None)
            if security_config and hasattr(security_config, "fingerprint"):
                try:
                    security_context["agent_fingerprint"] = (
                        security_config.fingerprint.to_dict()
                    )
                except AttributeError:
                    pass

        # Add task fingerprint if available
        if self.task and hasattr(self.task, "security_config"):
            security_config = getattr(self.task, "security_config", None)
            if security_config and hasattr(security_config, "fingerprint"):
                try:
                    security_context["task_fingerprint"] = (
                        security_config.fingerprint.to_dict()
                    )
                except AttributeError:
                    pass

        return arguments

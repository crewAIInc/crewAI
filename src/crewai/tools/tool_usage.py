import ast
from difflib import SequenceMatcher
from textwrap import dedent
from typing import Any, List, Union

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from crewai.agents.tools_handler import ToolsHandler
from crewai.telemetry import Telemetry
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.utilities import I18N, Converter, ConverterError, Printer

try:
    import agentops
except ImportError:
    agentops = None

OPENAI_BIGGER_MODELS = ["gpt-4"]


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
        tools_handler: ToolsHandler,
        tools: List[BaseTool],
        original_tools: List[Any],
        tools_description: str,
        tools_names: str,
        task: Any,
        function_calling_llm: Any,
        agent: Any,
        action: Any,
    ) -> None:
        self._i18n: I18N = I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remember_format_after_usages: int = 3
        self.agent = agent
        self.tools_description = tools_description
        self.tools_names = tools_names
        self.tools_handler = tools_handler
        self.original_tools = original_tools
        self.tools = tools
        self.task = task
        self.action = action
        self.function_calling_llm = function_calling_llm

        # Set the maximum parsing attempts for bigger models
        if (isinstance(self.function_calling_llm, ChatOpenAI)) and (
            self.function_calling_llm.openai_api_base is None
        ):
            if self.function_calling_llm.model_name in OPENAI_BIGGER_MODELS:
                self._max_parsing_attempts = 2
                self._remember_format_after_usages = 4

    def parse(self, tool_string: str):
        """Parse the tool string and return the tool calling."""
        return self._tool_calling(tool_string)

    def use(
        self, calling: Union[ToolCalling, InstructorToolCalling], tool_string: str
    ) -> str:
        if isinstance(calling, ToolUsageErrorException):
            error = calling.message
            self._printer.print(content=f"\n\n{error}\n", color="red")
            self.task.increment_tools_errors()
            return error

        # BUG? The code below seems to be unreachable
        try:
            tool = self._select_tool(calling.tool_name)
        except Exception as e:
            error = getattr(e, "message", str(e))
            self.task.increment_tools_errors()
            self._printer.print(content=f"\n\n{error}\n", color="red")
            return error
        return f"{self._use(tool_string=tool_string, tool=tool, calling=calling)}"  # type: ignore # BUG?: "_use" of "ToolUsage" does not return a value (it only ever returns None)

    def _use(
        self,
        tool_string: str,
        tool: BaseTool,
        calling: Union[ToolCalling, InstructorToolCalling],
    ) -> str:  # TODO: Fix this return type
        tool_event = agentops.ToolEvent(name=calling.tool_name) if agentops else None
        if self._check_tool_repeated_usage(calling=calling):  # type: ignore # _check_tool_repeated_usage of "ToolUsage" does not return a value (it only ever returns None)
            try:
                result = self._i18n.errors("task_repeated_usage").format(
                    tool_names=self.tools_names
                )
                self._printer.print(content=f"\n\n{result}\n", color="purple")
                self._telemetry.tool_repeated_usage(
                    llm=self.function_calling_llm,
                    tool_name=tool.name,
                    attempts=self._run_attempts,
                )
                result = self._format_result(result=result)  # type: ignore #  "_format_result" of "ToolUsage" does not return a value (it only ever returns None)
                return result  # type: ignore # Fix the reutrn type of this function

            except Exception:
                self.task.increment_tools_errors()

        result = None  # type: ignore # Incompatible types in assignment (expression has type "None", variable has type "str")

        if self.tools_handler.cache:
            result = self.tools_handler.cache.read(  # type: ignore # Incompatible types in assignment (expression has type "str | None", variable has type "str")
                tool=calling.tool_name, input=calling.arguments
            )

        original_tool = next(
            (ot for ot in self.original_tools if ot.name == tool.name), None
        )

        if result is None:  #! finecwg: if not result --> if result is None
            try:
                if calling.tool_name in [
                    "Delegate work to coworker",
                    "Ask question to coworker",
                ]:
                    self.task.increment_delegations()

                if calling.arguments:
                    try:
                        acceptable_args = tool.args_schema.schema()["properties"].keys()  # type: ignore # Item "None" of "type[BaseModel] | None" has no attribute "schema"
                        arguments = {
                            k: v
                            for k, v in calling.arguments.items()
                            if k in acceptable_args
                        }
                        result = tool.invoke(input=arguments)
                    except Exception:
                        arguments = calling.arguments
                        result = tool.invoke(input=arguments)
                else:
                    result = tool.invoke(input={})
            except Exception as e:
                self._run_attempts += 1
                if self._run_attempts > self._max_parsing_attempts:
                    self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                    error_message = self._i18n.errors("tool_usage_exception").format(
                        error=e, tool=tool.name, tool_inputs=tool.description
                    )
                    error = ToolUsageErrorException(
                        f'\n{error_message}.\nMoving on then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                    ).message
                    self.task.increment_tools_errors()
                    self._printer.print(content=f"\n\n{error_message}\n", color="red")
                    return error  # type: ignore # No return value expected

                self.task.increment_tools_errors()
                if agentops:
                    agentops.record(
                        agentops.ErrorEvent(exception=e, trigger_event=tool_event)
                    )
                return self.use(calling=calling, tool_string=tool_string)  # type: ignore # No return value expected

            if self.tools_handler:
                should_cache = True
                if (
                    hasattr(original_tool, "cache_function")
                    and original_tool.cache_function  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                ):
                    should_cache = original_tool.cache_function(  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
                        calling.arguments, result
                    )

                self.tools_handler.on_tool_use(
                    calling=calling, output=result, should_cache=should_cache
                )

        self._printer.print(content=f"\n\n{result}\n", color="purple")
        if agentops:
            agentops.record(tool_event)
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

        if (
            hasattr(original_tool, "result_as_answer")
            and original_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "cache_function"
        ):
            result_as_answer = original_tool.result_as_answer  # type: ignore # Item "None" of "Any | None" has no attribute "result_as_answer"
            data["result_as_answer"] = result_as_answer

        self.agent.tools_results.append(data)

        return result  # type: ignore # No return value expected

    def _format_result(self, result: Any) -> None:
        self.task.used_tools += 1
        if self._should_remember_format():  # type: ignore # "_should_remember_format" of "ToolUsage" does not return a value (it only ever returns None)
            result = self._remember_format(result=result)  # type: ignore # "_remember_format" of "ToolUsage" does not return a value (it only ever returns None)
        return result

    def _should_remember_format(self) -> None:
        return self.task.used_tools % self._remember_format_after_usages == 0

    def _remember_format(self, result: str) -> None:
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result  # type: ignore # No return value expected

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> None:
        if not self.tools_handler:
            return False  # type: ignore # No return value expected
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (  # type: ignore # No return value expected
                calling.arguments == last_tool_usage.arguments
            )

    def _select_tool(self, tool_name: str) -> BaseTool:
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
        self.task.increment_tools_errors()
        if tool_name and tool_name != "":
            raise Exception(
                f"Action '{tool_name}' don't exist, these are the only available Actions:\n {self.tools_description}"
            )
        else:
            raise Exception(
                f"I forgot the Action name, these are the only available Actions: {self.tools_description}"
            )

    def _render(self) -> str:
        """Render the tool name and description in plain text."""
        descriptions = []
        for tool in self.tools:
            args = {
                k: {k2: v2 for k2, v2 in v.items() if k2 in ["description", "type"]}
                for k, v in tool.args.items()
            }
            descriptions.append(
                "\n".join(
                    [
                        f"Tool Name: {tool.name.lower()}",
                        f"Tool Description: {tool.description}",
                        f"Tool Arguments: {args}",
                    ]
                )
            )
        return "\n--\n".join(descriptions)

    def _is_gpt(self, llm) -> bool:
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base is None

    def _tool_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling]:
        try:
            if self.function_calling_llm:
                model = (
                    InstructorToolCalling
                    if self._is_gpt(self.function_calling_llm)
                    else ToolCalling
                )
                converter = Converter(
                    text=f"Only tools available:\n###\n{self._render()}\n\nReturn a valid schema for the tool, the tool name must be exactly equal one of the options, use this text to inform the valid output schema:\n\n{tool_string}```",
                    llm=self.function_calling_llm,
                    model=model,
                    instructions=dedent(
                        """\
              The schema should have the following structure, only two keys:
              - tool_name: str
              - arguments: dict (with all arguments being passed)

              Example:
              {"tool_name": "tool name", "arguments": {"arg_name1": "value", "arg_name2": 2}}""",
                    ),
                    max_attempts=1,
                )
                calling = converter.to_pydantic()

                if isinstance(calling, ConverterError):
                    raise calling
            else:
                tool_name = self.action.tool
                tool = self._select_tool(tool_name)
                try:
                    tool_input = self._validate_tool_input(self.action.tool_input)
                    arguments = ast.literal_eval(tool_input)
                except Exception:
                    return ToolUsageErrorException(  # type: ignore # Incompatible return value type (got "ToolUsageErrorException", expected "ToolCalling | InstructorToolCalling")
                        f'{self._i18n.errors("tool_arguments_error")}'
                    )
                if not isinstance(arguments, dict):
                    return ToolUsageErrorException(  # type: ignore # Incompatible return value type (got "ToolUsageErrorException", expected "ToolCalling | InstructorToolCalling")
                        f'{self._i18n.errors("tool_arguments_error")}'
                    )
                calling = ToolCalling(  # type: ignore # Unexpected keyword argument "log" for "ToolCalling"
                    tool_name=tool.name,
                    arguments=arguments,
                    log=tool_string,
                )
        except Exception as e:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.function_calling_llm)
                self.task.increment_tools_errors()
                self._printer.print(content=f"\n\n{e}\n", color="red")
                return ToolUsageErrorException(  # type: ignore # Incompatible return value type (got "ToolUsageErrorException", expected "ToolCalling | InstructorToolCalling")
                    f'{self._i18n.errors("tool_usage_error").format(error=e)}\nMoving on then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                )
            return self._tool_calling(tool_string)

        return calling

    def _validate_tool_input(self, tool_input: str) -> str:
        try:
            ast.literal_eval(tool_input)
            return tool_input
        except Exception:
            # Clean and ensure the string is properly enclosed in braces
            tool_input = tool_input.strip()
            if not tool_input.startswith("{"):
                tool_input = "{" + tool_input
            if not tool_input.endswith("}"):
                tool_input += "}"

            # Manually split the input into key-value pairs
            entries = tool_input.strip("{} ").split(",")
            formatted_entries = []

            for entry in entries:
                if ":" not in entry:
                    continue  # Skip malformed entries
                key, value = entry.split(":", 1)

                # Remove extraneous white spaces and quotes, replace single quotes
                key = key.strip().strip('"').replace("'", '"')
                value = value.strip()

                # Handle replacement of single quotes at the start and end of the value string
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]  # Remove single quotes
                    value = (
                        '"' + value.replace('"', '\\"') + '"'
                    )  # Re-encapsulate with double quotes
                elif value.isdigit():  # Check if value is a digit, hence integer
                    formatted_value = value
                elif value.lower() in [
                    "true",
                    "false",
                    "null",
                ]:  # Check for boolean and null values
                    formatted_value = value.lower()
                else:
                    # Assume the value is a string and needs quotes
                    formatted_value = '"' + value.replace('"', '\\"') + '"'

                # Rebuild the entry with proper quoting
                formatted_entry = f'"{key}": {formatted_value}'
                formatted_entries.append(formatted_entry)

            # Reconstruct the JSON string
            new_json_string = "{" + ", ".join(formatted_entries) + "}"
            return new_json_string

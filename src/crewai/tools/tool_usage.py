import ast
from textwrap import dedent
from typing import Any, List, Union

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from crewai.agents.tools_handler import ToolsHandler
from crewai.telemetry import Telemetry
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.utilities import I18N, Converter, ConverterError, Printer

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
        action: Any,
    ) -> None:
        self._i18n: I18N = I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remember_format_after_usages: int = 3
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
            self.function_calling_llm.openai_api_base == None
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
        try:
            tool = self._select_tool(calling.tool_name)
        except Exception as e:
            error = getattr(e, "message", str(e))
            self.task.increment_tools_errors()
            self._printer.print(content=f"\n\n{error}\n", color="red")
            return error
        return f"{self._use(tool_string=tool_string, tool=tool, calling=calling)}"

    def _use(
        self,
        tool_string: str,
        tool: BaseTool,
        calling: Union[ToolCalling, InstructorToolCalling],
    ) -> None:
        if self._check_tool_repeated_usage(calling=calling):
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
                result = self._format_result(result=result)
                return result
            except Exception:
                self.task.increment_tools_errors()

        result = None

        if self.tools_handler.cache:
            result = self.tools_handler.cache.read(
                tool=calling.tool_name, input=calling.arguments
            )

        if not result:
            try:
                if calling.tool_name in [
                    "Delegate work to co-worker",
                    "Ask question to co-worker",
                ]:
                    self.task.increment_delegations()

                if calling.arguments:
                    try:
                        acceptable_args = tool.args_schema.schema()["properties"].keys()
                        arguments = {
                            k: v
                            for k, v in calling.arguments.items()
                            if k in acceptable_args
                        }
                        result = tool._run(**arguments)
                    except Exception:
                        if tool.args_schema:
                            arguments = calling.arguments
                            result = tool._run(**arguments)
                        else:
                            arguments = calling.arguments.values()
                            result = tool._run(*arguments)
                else:
                    result = tool._run()
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
                    return error
                self.task.increment_tools_errors()
                return self.use(calling=calling, tool_string=tool_string)

            if self.tools_handler:
                should_cache = True
                original_tool = next(
                    (ot for ot in self.original_tools if ot.name == tool.name), None
                )
                if (
                    hasattr(original_tool, "cache_function")
                    and original_tool.cache_function
                ):
                    should_cache = original_tool.cache_function(
                        calling.arguments, result
                    )

                self.tools_handler.on_tool_use(
                    calling=calling, output=result, should_cache=should_cache
                )

        self._printer.print(content=f"\n\n{result}\n", color="purple")
        self._telemetry.tool_usage(
            llm=self.function_calling_llm,
            tool_name=tool.name,
            attempts=self._run_attempts,
        )
        result = self._format_result(result=result)
        return result

    def _format_result(self, result: Any) -> None:
        self.task.used_tools += 1
        if self._should_remember_format():
            result = self._remember_format(result=result)
        return result

    def _should_remember_format(self) -> None:
        return self.task.used_tools % self._remember_format_after_usages == 0

    def _remember_format(self, result: str) -> None:
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> None:
        if not self.tools_handler:
            return False
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (
                calling.arguments == last_tool_usage.arguments
            )

    def _select_tool(self, tool_name: str) -> BaseTool:
        for tool in self.tools:
            if tool.name.lower().strip() == tool_name.lower().strip():
                return tool
        self.task.increment_tools_errors()
        if tool_name and tool_name != "":
            raise Exception(
                f"Action '{tool_name}' don't exist, these are the only available Actions: {self.tools_description}"
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
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base == None

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
                    text=f"Only tools available:\n###\n{self._render()}\n\nReturn a valid schema for the tool, the tool name must be exactly equal one of the options, use this text to inform the valid ouput schema:\n\n{tool_string}```",
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
                    max_attemps=1,
                )
                calling = converter.to_pydantic()

                if isinstance(calling, ConverterError):
                    raise calling
            else:
                tool_name = self.action.tool
                tool = self._select_tool(tool_name)
                try:
                    arguments = ast.literal_eval(self.action.tool_input)
                except Exception:
                    return ToolUsageErrorException(
                        f'{self._i18n.errors("tool_arguments_error")}'
                    )
                if not isinstance(arguments, dict):
                    return ToolUsageErrorException(
                        f'{self._i18n.errors("tool_arguments_error")}'
                    )
                calling = ToolCalling(
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
                return ToolUsageErrorException(
                    f'{self._i18n.errors("tool_usage_error").format(error=e)}\nMoving on then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                )
            return self._tool_calling(tool_string)

        return calling

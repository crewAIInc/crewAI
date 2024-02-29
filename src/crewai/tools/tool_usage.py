from textwrap import dedent
from typing import Any, List, Union

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from crewai.agents.tools_handler import ToolsHandler
from crewai.telemtry import Telemetry
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.utilities import I18N, Converter, ConverterError, Printer

OPENAI_BIGGER_MODELS = ["gpt-4"]


class ToolUsageErrorException(Exception):
    """
    This is a custom exception class that is raised when there are errors in the usage of a tool.
    It takes a message as an argument which describes the error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ToolUsage:
    """
    This class represents the usage of a tool by an agent. It contains methods for parsing tool strings,
    using tools, and handling errors in tool usage.

    Attributes:
        task: The task that the agent is currently executing.
        tools_handler: An instance of the ToolsHandler class that manages the usage of tools.
        tools: A list of tools that are available for the agent to use.
        tools_description: A string that describes the tools that are available for the agent to use.
        tools_names: A string that contains the names of the tools that are available for the agent to use.
        llm: The language model that is used for the tool usage.
        function_calling_llm: The language model that is used for calling functions in the tool usage.
    """

    def __init__(
        self,
        tools_handler: ToolsHandler,
        tools: List[BaseTool],
        tools_description: str,
        tools_names: str,
        task: Any,
        llm: Any,
        function_calling_llm: Any,
    ) -> None:
        # Initialize various helper classes and variables
        self._i18n: I18N = I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remember_format_after_usages: int = 3
        self.tools_description = tools_description
        self.tools_names = tools_names
        self.tools_handler = tools_handler
        self.tools = tools
        self.task = task
        self.llm = function_calling_llm or llm

        # If the language model is a bigger model, adjust the maximum parsing attempts and remember format after usages
        if (isinstance(self.llm, ChatOpenAI)) and (self.llm.openai_api_base == None):
            if self.llm.model_name in OPENAI_BIGGER_MODELS:
                self._max_parsing_attempts = 2
                self._remember_format_after_usages = 4

    def parse(self, tool_string: str):
        """
        This method takes a tool string as input and returns a ToolCalling object.
        The tool string is parsed to extract the tool name and arguments.
        """
        return self._tool_calling(tool_string)

    def use(
        self, calling: Union[ToolCalling, InstructorToolCalling], tool_string: str
    ) -> str:
        """
        This method takes a ToolCalling object and a tool string as input and uses the tool.
        If the calling object is an instance of ToolUsageErrorException, it prints the error message and increments the tool errors count.
        If the tool cannot be selected, it prints the error message and increments the tool errors count.
        """
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
        return f"{self._use(tool_string=tool_string, tool=tool, calling=calling)}\n\n{self._i18n.slice('final_answer_format')}"

    def _use(
            self,
            tool_string: str,
            tool: BaseTool,
            calling: Union[ToolCalling, InstructorToolCalling],
        ) -> None:
            """
            This method takes a tool string, a BaseTool object, and a ToolCalling object as input and uses the tool.
            If the tool has been used repeatedly, it prints a warning message and sends telemetry data.
            If the tool result is not in the cache, it runs the tool and handles any exceptions that occur.
            If the tool result is in the cache, it prints the result and sends telemetry data.
            """
            if self._check_tool_repeated_usage(calling=calling):
                try:
                    result = self._i18n.errors("task_repeated_usage").format(
                        tool=calling.tool_name,
                        tool_input=", ".join(
                            [str(arg) for arg in calling.arguments.values()]
                        ),
                    )
                    self._printer.print(content=f"\n\n{result}\n", color="yellow")
                    self._telemetry.tool_repeated_usage(
                        llm=self.llm, tool_name=tool.name, attempts=self._run_attempts
                    )
                    result = self._format_result(result=result)
                    return result
                except Exception:
                    self.task.increment_tools_errors()

            result = self.tools_handler.cache.read(
                tool=calling.tool_name, input=calling.arguments
            )

            if not result:
                try:
                    # If the tool name is either "Delegate work to co-worker" or "Ask question to co-worker", increment the delegations count
                    if calling.tool_name in [
                        "Delegate work to co-worker",
                        "Ask question to co-worker",
                    ]:
                        self.task.increment_delegations()

                    # If there are arguments for the tool, run the tool with the arguments
                    # Otherwise, run the tool without any arguments
                    if calling.arguments:
                        result = tool._run(**calling.arguments)
                    else:
                        result = tool._run()
                except Exception as e:
                    self._run_attempts += 1
                    if self._run_attempts > self._max_parsing_attempts:
                        self._telemetry.tool_usage_error(llm=self.llm)
                        error_message = self._i18n.errors("tool_usage_exception").format(
                            error=e
                        )
                        error = ToolUsageErrorException(
                            f'\n{error_message}.\nMoving one then. {self._i18n.slice("format").format(tool_names=self.tools_names)}'
                        ).message
                        self.task.increment_tools_errors()
                        self._printer.print(content=f"\n\n{error_message}\n", color="red")
                        return error
                    self.task.increment_tools_errors()
                    return self.use(calling=calling, tool_string=tool_string)

                self.tools_handler.on_tool_use(calling=calling, output=result)

            self._printer.print(content=f"\n\n{result}\n", color="yellow")
            self._telemetry.tool_usage(
                llm=self.llm, tool_name=tool.name, attempts=self._run_attempts
            )
            result = self._format_result(result=result)
            return result

    def _format_result(self, result: Any) -> None:
        """
        This method takes a result as input and formats it.
        It increments the used tools count and checks if the format should be remembered.
        If the format should be remembered, it remembers the format.
        """
        self.task.used_tools += 1
        if self._should_remember_format():
            result = self._remember_format(result=result)
        return result

    def _should_remember_format(self) -> None:
        """
        This method checks if the format should be remembered.
        It returns True if the used tools count is a multiple of the remember format after usages count, and False otherwise.
        """
        return self.task.used_tools % self._remember_format_after_usages == 0

    def _remember_format(self, result: str) -> None:
        """
        This method takes a result string as input and appends a formatted string containing tool descriptions and names.
        The formatted string is then returned.
        """
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> None:
        """
        This method checks if the last used tool is the same as the current tool being called.
        It takes a ToolCalling object as input and returns a boolean value.
        """
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (
                calling.arguments == last_tool_usage.arguments
            )

    def _select_tool(self, tool_name: str) -> BaseTool:
        """
        This method selects a tool based on the tool name.
        It takes a tool name as input and returns a BaseTool object.
        If the tool is not found, it increments the tool errors count and raises an exception.
        """
        for tool in self.tools:
            if tool.name.lower().strip() == tool_name.lower().strip():
                return tool
        self.task.increment_tools_errors()
        raise Exception(f"Tool '{tool_name}' not found.")

    def _render(self) -> str:
        """
        This method renders the tool name and description in plain text.
        It returns a string containing the tool name, description, and arguments.
        """
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
        """
        This method checks if the language model is a GPT model.
        It takes a language model as input and returns a boolean value.
        """
        return isinstance(llm, ChatOpenAI) and llm.openai_api_base == None

    def _tool_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling]:
        """
        This method converts a tool string into a ToolCalling object.
        It takes a tool string as input and returns a ToolCalling object.
        If the conversion fails, it increments the run attempts count and handles the exception.
        """
        try:
            model = InstructorToolCalling if self._is_gpt(self.llm) else ToolCalling
            converter = Converter(
                text=f"Only tools available:\n###\n{self._render()}\n\nReturn a valid schema for the tool, the tool name must be exactly equal one of the options, use this text to inform the valid ouput schema:\n\n{tool_string}```",
                llm=self.llm,
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
        except Exception as e:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.llm)
                self.task.increment_tools_errors()
                self._printer.print(content=f"\n\n{e}\n", color="red")
                return ToolUsageErrorException(
                    f'{self._i18n.errors("tool_usage_error")}\n{self._i18n.slice("format").format(tool_names=self.tools_names)}'
                )
            return self._tool_calling(tool_string)

        return calling

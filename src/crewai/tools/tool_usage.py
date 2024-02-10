from typing import Any, List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from crewai.agents.tools_handler import ToolsHandler
from crewai.telemtry import Telemetry
from crewai.tools.tool_calling import ToolCalling
from crewai.utilities import I18N, Printer


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
        tools_description: Description of the tools available for the agent.
        tools_names: Names of the tools available for the agent.
        llm: Language model to be used for the tool usage.
    """

    def __init__(
        self,
        tools_handler: ToolsHandler,
        tools: List[BaseTool],
        tools_description: str,
        tools_names: str,
        task: Any,
        llm: Any,
    ) -> None:
        self._i18n: I18N = I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 3
        self._remeber_format_after_usages: int = 3
        self.tools_description = tools_description
        self.tools_names = tools_names
        self.tools_handler = tools_handler
        self.tools = tools
        self.task = task
        self.llm = llm

    def use(self, tool_string: str):
        calling = self._tool_calling(tool_string)
        if isinstance(calling, ToolUsageErrorException):
            error = calling.message
            self._printer.print(content=f"\n\n{error}\n", color="yellow")
            return error
        tool = self._select_tool(calling.function_name)
        return self._use(tool_string=tool_string, tool=tool, calling=calling)

    def _use(self, tool_string: str, tool: BaseTool, calling: ToolCalling) -> None:
        try:
            if self._check_tool_repeated_usage(calling=calling):
                result = self._i18n.errors("task_repeated_usage").format(
                    tool=calling.function_name, tool_input=calling.arguments
                )
            else:
                self.tools_handler.on_tool_start(calling=calling)

                result = self.tools_handler.cache.read(
                    tool=calling.function_name, input=calling.arguments
                )

                if not result:
                    result = tool._run(**calling.arguments)
                    self.tools_handler.on_tool_end(calling=calling, output=result)

            self._printer.print(content=f"\n\n{result}\n", color="yellow")
            self._telemetry.tool_usage(
                llm=self.llm, tool_name=tool.name, attempts=self._run_attempts
            )

            result = self._format_result(result=result)
            return result
        except Exception:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.llm)
                return ToolUsageErrorException(
                    self._i18n.errors("tool_usage_error")
                ).message
            return self.use(tool_string=tool_string)

    def _format_result(self, result: Any) -> None:
        self.task.used_tools += 1
        if self._should_remember_format():
            result = self._remember_format(result=result)
        return result

    def _should_remember_format(self) -> None:
        return self.task.used_tools % self._remeber_format_after_usages == 0

    def _remember_format(self, result: str) -> None:
        result = str(result)
        result += "\n\n" + self._i18n.slice("tools").format(
            tools=self.tools_description, tool_names=self.tools_names
        )
        return result

    def _check_tool_repeated_usage(self, calling: ToolCalling) -> None:
        if last_tool_usage := self.tools_handler.last_used_tool:
            return calling == last_tool_usage

    def _select_tool(self, tool_name: str) -> BaseTool:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise Exception(f"Tool '{tool_name}' not found.")

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
                        f"Funtion Name: {tool.name}",
                        f"Funtion attributes: {args}",
                        f"Description: {tool.description}",
                    ]
                )
            )
        return "\n--\n".join(descriptions)

    def _tool_calling(self, tool_string: str) -> ToolCalling:
        try:
            parser = PydanticOutputParser(pydantic_object=ToolCalling)
            prompt = PromptTemplate(
                template="Return a valid schema for the one tool you must use with its arguments and values.\n\nTools available:\n\n{available_tools}\n\nUse this text to inform a valid ouput schema:\n{tool_string}\n\n{format_instructions}\n```",
                input_variables=["tool_string"],
                partial_variables={
                    "available_tools": self._render(),
                    "format_instructions": parser.get_format_instructions(),
                },
            )
            chain = prompt | self.llm | parser
            calling = chain.invoke({"tool_string": tool_string})

        except Exception:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=self.llm)
                return ToolUsageErrorException(self._i18n.errors("tool_usage_error"))
            return self._tool_calling(tool_string)

        return calling

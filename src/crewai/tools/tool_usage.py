from typing import Any, List, Union

import instructor
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from crewai.agents.tools_handler import ToolsHandler
from crewai.telemtry import Telemetry
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.tools.tool_output_parser import ToolOutputParser
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
        function_calling_llm: Any,
    ) -> None:
        self._i18n: I18N = I18N()
        self._printer: Printer = Printer()
        self._telemetry: Telemetry = Telemetry()
        self._run_attempts: int = 1
        self._max_parsing_attempts: int = 2
        self._remeber_format_after_usages: int = 3
        self.tools_description = tools_description
        self.tools_names = tools_names
        self.tools_handler = tools_handler
        self.tools = tools
        self.task = task
        self.llm = llm
        self.function_calling_llm = function_calling_llm

    def use(self, tool_string: str):
        calling = self._tool_calling(tool_string)
        if isinstance(calling, ToolUsageErrorException):
            error = calling.message
            self._printer.print(content=f"\n\n{error}\n", color="yellow")
            return error
        try:
            tool = self._select_tool(calling.tool_name)
        except Exception as e:
            error = getattr(e, "message", str(e))
            self._printer.print(content=f"\n\n{error}\n", color="yellow")
            return error
        return self._use(tool_string=tool_string, tool=tool, calling=calling)

    def _use(
        self,
        tool_string: str,
        tool: BaseTool,
        calling: Union[ToolCalling, InstructorToolCalling],
    ) -> None:
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
                pass

        self.tools_handler.on_tool_start(calling=calling)

        result = self.tools_handler.cache.read(
            tool=calling.tool_name, input=calling.arguments
        )

        if not result:
            try:
                result = tool._run(**calling.arguments)
            except Exception as e:
                self._run_attempts += 1
                if self._run_attempts > self._max_parsing_attempts:
                    self._telemetry.tool_usage_error(llm=self.llm)
                    return ToolUsageErrorException(
                        self._i18n.errors("tool_usage_exception").format(error=e)
                    ).message
                return self.use(tool_string=tool_string)

            self.tools_handler.on_tool_end(calling=calling, output=result)

        self._printer.print(content=f"\n\n{result}\n", color="yellow")
        self._telemetry.tool_usage(
            llm=self.llm, tool_name=tool.name, attempts=self._run_attempts
        )

        result = self._format_result(result=result)
        return result

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

    def _check_tool_repeated_usage(
        self, calling: Union[ToolCalling, InstructorToolCalling]
    ) -> None:
        if last_tool_usage := self.tools_handler.last_used_tool:
            return (calling.tool_name == last_tool_usage.tool_name) and (
                calling.arguments == last_tool_usage.arguments
            )

    def _select_tool(self, tool_name: str) -> BaseTool:
        for tool in self.tools:
            if tool.name.lower().strip() == tool_name.lower().strip():
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
                        f"Tool Name: {tool.name.lower()}",
                        f"Tool Description: {tool.description}",
                        f"Tool Arguments: {args}",
                    ]
                )
            )
        return "\n--\n".join(descriptions)

    def _tool_calling(
        self, tool_string: str
    ) -> Union[ToolCalling, InstructorToolCalling]:
        try:
            tool_string = tool_string.replace(
                "Thought: Do I need to use a tool? Yes", ""
            )
            tool_string = tool_string.replace("Action:", "Tool Name:")
            tool_string = tool_string.replace("Action Input:", "Tool Arguments:")

            llm = self.function_calling_llm or self.llm

            if (isinstance(llm, ChatOpenAI)) and (llm.openai_api_base == None):
                client = instructor.patch(
                    llm.client._client,
                    mode=instructor.Mode.FUNCTIONS,
                )
                calling = client.chat.completions.create(
                    model=llm.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """
                                The schema should have the following structure, only two key:
                                - tool_name: str
                                - arguments: dict (with all arguments being passed)

                                Example:
                                {"tool_name": "tool_name", "arguments": {"arg_name1": "value", "arg_name2": 2}}
                            """,
                        },
                        {
                            "role": "user",
                            "content": f"Tools available:\n\n{self._render()}\n\nReturn a valid schema for the tool, use this text to inform a valid ouput schema:\n{tool_string}```",
                        },
                    ],
                    response_model=InstructorToolCalling,
                )
            else:
                parser = ToolOutputParser(pydantic_object=ToolCalling)
                prompt = PromptTemplate(
                    template="Tools available:\n\n{available_tools}\n\nReturn a valid schema for the tool, use this text to inform a valid ouput schema:\n{tool_string}\n\n{format_instructions}\n```",
                    input_variables=["tool_string"],
                    partial_variables={
                        "available_tools": self._render(),
                        "format_instructions": """
                        The schema should have the following structure, only two key:
                        - tool_name: str
                        - arguments: dict (with all arguments being passed)

                        Example:
                        {"tool_name": "tool_name", "arguments": {"arg_name1": "value", "arg_name2": 2}}
                        """,
                    },
                )
                chain = prompt | llm | parser
                calling = chain.invoke({"tool_string": tool_string})

        except Exception:
            self._run_attempts += 1
            if self._run_attempts > self._max_parsing_attempts:
                self._telemetry.tool_usage_error(llm=llm)
                return ToolUsageErrorException(self._i18n.errors("tool_usage_error"))
            return self._tool_calling(tool_string)

        return calling

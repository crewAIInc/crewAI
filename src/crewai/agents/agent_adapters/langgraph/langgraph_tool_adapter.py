import inspect
from typing import Any, List, Optional

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools.base_tool import BaseTool


class LangGraphToolAdapter(BaseToolAdapter):
    """Adapts CrewAI tools to LangGraph agent tool compatible format"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.original_tools = tools or []
        self.converted_tools = []

    def configure_tools(self, tools: List[BaseTool]) -> None:
        """
        Configure and convert CrewAI tools to LangGraph-compatible format.
        LangGraph expects tools in langchain_core.tools format.
        """
        from langchain_core.tools import BaseTool, StructuredTool

        converted_tools = []
        if self.original_tools:
            all_tools = tools + self.original_tools
        else:
            all_tools = tools
        for tool in all_tools:
            if isinstance(tool, BaseTool):
                converted_tools.append(tool)
                continue

            sanitized_name = self.sanitize_tool_name(tool.name)

            async def tool_wrapper(*args, tool=tool, **kwargs):
                output = None
                if len(args) > 0 and isinstance(args[0], str):
                    output = tool.run(args[0])
                elif "input" in kwargs:
                    output = tool.run(kwargs["input"])
                else:
                    output = tool.run(**kwargs)

                if inspect.isawaitable(output):
                    result = await output
                else:
                    result = output
                return result

            converted_tool = StructuredTool(
                name=sanitized_name,
                description=tool.description,
                func=tool_wrapper,
                args_schema=tool.args_schema,
            )

            converted_tools.append(converted_tool)

        self.converted_tools = converted_tools

    def tools(self) -> List[Any]:
        return self.converted_tools or []

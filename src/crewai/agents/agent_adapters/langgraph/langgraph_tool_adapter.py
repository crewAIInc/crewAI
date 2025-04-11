from typing import Any, List, Optional

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools.base_tool import BaseTool


class LangGraphToolAdapter(BaseToolAdapter):
    """Adapts CrewAI tools to LangGraph agent tool compatible format"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.original_tools = tools or []

    def configure_tools(self, tools: List[BaseTool]) -> None:
        """
        Configure and convert CrewAI tools to LangGraph-compatible format.
        LangGraph expects tools in langchain_core.tools format.
        """
        from langchain_core.tools import StructuredTool

        self.tools = tools
        self.converted_tools = []
        if self.original_tools:
            all_tools = tools + self.original_tools
        else:
            all_tools = tools
        for tool in all_tools:
            # Create a wrapper function that matches LangGraph's expected format
            def tool_wrapper(*args, tool=tool, **kwargs):
                # Extract inputs based on the tool's schema
                if len(args) > 0 and isinstance(args[0], str):
                    return tool.run(args[0])
                elif "input" in kwargs:
                    return tool.run(kwargs["input"])
                else:
                    return tool.run(**kwargs)

            converted_tool = StructuredTool(
                name=tool.name.replace(" ", "_"),
                description=tool.description,
                func=tool_wrapper,
                args_schema=tool.args_schema,
            )

            self.converted_tools.append(converted_tool)

    def all_tools(self) -> List[Any]:
        return self.converted_tools

class LangGraphToolAdapter:
    """Adapts CrewAI tools to LangGraph-compatible format"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.tools = tools or []
        self.converted_tools = []

    def configure_tools(self, tools: List[BaseTool]) -> None:
        """Convert CrewAI tools to LangGraph tools"""
        self.tools = tools
        self.converted_tools = self._convert_tools(tools)

    def _convert_tools(self, tools: List[BaseTool]) -> List[Any]:
        """
        Convert CrewAI tools to LangGraph-compatible tools
        LangGraph expects tools in langchain_core.tools format
        """
        from langchain_core.tools import Tool

        converted_tools = []

        for tool in tools:
            # Create a wrapper function that matches LangGraph's expected format
            def tool_wrapper(*args, tool=tool, **kwargs):
                # Extract inputs based on the tool's schema
                if len(args) > 0 and isinstance(args[0], str):
                    return tool.run(args[0])
                elif "input" in kwargs:
                    return tool.run(kwargs["input"])
                else:
                    return tool.run(**kwargs)

            # Create a LangChain Tool
            converted_tool = Tool(
                name=tool.name, description=tool.description, func=tool_wrapper
            )

            converted_tools.append(converted_tool)

        return converted_tools

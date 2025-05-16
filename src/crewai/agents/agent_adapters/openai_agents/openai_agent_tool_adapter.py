import inspect
from typing import Any, List, Optional

from agents import FunctionTool, Tool

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools import BaseTool


class OpenAIAgentToolAdapter(BaseToolAdapter):
    """Adapter for OpenAI Assistant tools"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.original_tools = tools or []

    def configure_tools(self, tools: List[BaseTool]) -> None:
        """Configure tools for the OpenAI Assistant"""
        if self.original_tools:
            all_tools = tools + self.original_tools
        else:
            all_tools = tools
        if all_tools:
            self.converted_tools = self._convert_tools_to_openai_format(all_tools)

    def _convert_tools_to_openai_format(
        self, tools: Optional[List[BaseTool]]
    ) -> List[Tool]:
        """Convert CrewAI tools to OpenAI Assistant tool format"""
        if not tools:
            return []

        def sanitize_tool_name(name: str) -> str:
            """Convert tool name to match OpenAI's required pattern"""
            import re

            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()
            return sanitized

        def create_tool_wrapper(tool: BaseTool):
            """Create a wrapper function that handles the OpenAI function tool interface"""

            async def wrapper(context_wrapper: Any, arguments: Any) -> Any:
                # Get the parameter name from the schema
                param_name = list(
                    tool.args_schema.model_json_schema()["properties"].keys()
                )[0]

                # Handle different argument types
                if isinstance(arguments, dict):
                    args_dict = arguments
                elif isinstance(arguments, str):
                    try:
                        import json

                        args_dict = json.loads(arguments)
                    except json.JSONDecodeError:
                        args_dict = {param_name: arguments}
                else:
                    args_dict = {param_name: str(arguments)}

                # Run the tool with the processed arguments
                output = tool._run(**args_dict)

                # Await if the tool returned a coroutine
                if inspect.isawaitable(output):
                    result = await output
                else:
                    result = output

                # Ensure the result is JSON serializable
                if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                    return result
                return str(result)

            return wrapper

        openai_tools = []
        for tool in tools:
            schema = tool.args_schema.model_json_schema()

            schema.update({"additionalProperties": False, "type": "object"})

            openai_tool = FunctionTool(
                name=sanitize_tool_name(tool.name),
                description=tool.description,
                params_json_schema=schema,
                on_invoke_tool=create_tool_wrapper(tool),
            )
            openai_tools.append(openai_tool)

        return openai_tools

import inspect
import re
from typing import Any, List, Optional

from azure.ai.agents.models import FunctionTool
from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools import BaseTool

class FoundryAgentToolAdapter(BaseToolAdapter):
    """Adapter for Foundry Assistant tools"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.original_tools = tools or []

    def configure_tools(self, tools: List[BaseTool]) -> None:
        all_tools = tools + (self.original_tools or [])

        def make_wrapped_func(tool: BaseTool):
            def wrapped(**kwargs):
                return tool.run(**kwargs)

            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool.name).lower()
            wrapped.__name__ = safe_name
            wrapped.__doc__ = tool.description or f"{tool.name} tool"

            wrapped.__annotations__ = {
                name: field.annotation or Any
                for name, field in tool.args_schema.model_fields.items()
            }

            return wrapped

        self.converted_tools = FunctionTool(
            functions={make_wrapped_func(t) for t in all_tools}
        )

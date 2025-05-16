import inspect
from typing import Any, List, Optional

from azure.ai.projects.models import (
    CodeInterpreterTool,
    FileSearchToolDefinition,
    AzureAISearchToolDefinition,
    BingCustomSearchToolDefinition,
    AzureFunctionToolDefinition,
)

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools import BaseTool
from azure.ai.projects.models import FunctionToolDefinition


class FoundryAgentToolAdapter(BaseToolAdapter):
    """Adapter for Foundry Assistant tools"""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        # Instantiate what we can now
        #code_interpreter = CodeInterpreterTool()

        # Stub placeholders for unconfigured tools
        #file_search = FileSearchToolDefinition(name="file-search")
        #azure_search = AzureAISearchToolDefinition(name="azure-search")
        #bing_search = BingCustomSearchToolDefinition(name="bing-search")
        #azure_fn = AzureFunctionToolDefinition(name="azure-function")

        # Aggregate them
        #built_in_tools = (
        #    code_interpreter.definitions +
        #    [file_search, azure_search, bing_search, azure_fn]
        #)

        self.original_tools = (tools or []) #+ built_in_tools
        #self.tool_resources = code_interpreter.resources

    def configure_tools(self, tools: List[BaseTool]) -> None:
        """Configure tools for the Foundry Assistant"""
        if self.original_tools:
            all_tools = tools + self.original_tools
        else:
            all_tools = tools
        if all_tools:
            self.converted_tools = self._convert_tools_to_foundry_format(all_tools)

    def _convert_tools_to_foundry_format(
        self, tools: Optional[List[BaseTool]]
    ) -> List[Any]:
        """
        Converts a list of BaseTool instances into Azure Foundry-compatible FunctionToolDefinitions.
        This version assumes you're using the standard SDK with no direct callable FunctionTool.
        """

        if not tools:
            return []

        foundry_tools = []
        for tool in tools:
            tool._set_args_schema()
            schema = tool.args_schema.model_json_schema()

            foundry_tool = FunctionToolDefinition(
                #name=tool.name,
                #description=tool.description,
                #parameters=schema
            )
            foundry_tools.append(foundry_tool)

        return foundry_tools


from typing import Any, List, Optional, Dict, Callable, Union

from pydantic import Field, model_validator

from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool


class ToolWithInstruction(BaseTool):
    """A wrapper for tools that adds specific usage instructions.
    
    This allows users to provide specific instructions on when and how to use a tool,
    without having to include these instructions in the agent's backstory.
    
    Attributes:
        tool: The tool to wrap
        instructions: Specific instructions about when and how to use this tool
        name: Name of the tool (inherited from the wrapped tool)
        description: Description of the tool (inherited from the wrapped tool with instructions)
    """
    
    name: str = Field(default="", description="Name of the tool")
    description: str = Field(default="", description="Description of the tool")
    tool: BaseTool = Field(description="The tool to wrap")
    instructions: str = Field(description="Instructions about when and how to use this tool")
    
    @model_validator(mode="after")
    def set_tool_attributes(self) -> "ToolWithInstruction":
        """Set attributes from the wrapped tool."""
        self.name = self.tool.name
        self.description = f"{self.tool.description}\nInstructions: {self.instructions}"
        self.args_schema = self.tool.args_schema
        return self
    
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped tool."""
        return self.tool._run(*args, **kwargs)
    
    def to_structured_tool(self) -> CrewStructuredTool:
        """Convert this tool to a CrewStructuredTool instance."""
        structured_tool = self.tool.to_structured_tool()
        
        structured_tool.description = f"{structured_tool.description}\nInstructions: {self.instructions}"
        
        return structured_tool

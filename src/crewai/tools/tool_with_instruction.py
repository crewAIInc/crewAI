from typing import Any, List, Optional, Dict, Callable, Union, ClassVar

from pydantic import Field, model_validator, field_validator, ConfigDict

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
    
    MAX_INSTRUCTION_LENGTH: ClassVar[int] = 2000
    
    name: str = Field(default="", description="Name of the tool")
    description: str = Field(default="", description="Description of the tool")
    tool: BaseTool = Field(description="The tool to wrap")
    instructions: str = Field(description="Instructions about when and how to use this tool")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator("instructions")
    @classmethod
    def validate_instructions(cls, value: str) -> str:
        """Validate that instructions are not empty and not too long.
        
        Args:
            value: The instructions string to validate
            
        Returns:
            str: The validated and sanitized instructions
            
        Raises:
            ValueError: If instructions are empty or exceed maximum length
        """
        if not value or not value.strip():
            raise ValueError("Instructions cannot be empty")
        
        if len(value) > cls.MAX_INSTRUCTION_LENGTH:
            raise ValueError(
                f"Instructions exceed maximum length of {cls.MAX_INSTRUCTION_LENGTH} characters"
            )
        
        return value.strip()
    
    @model_validator(mode="after")
    def set_tool_attributes(self) -> "ToolWithInstruction":
        """Sets name, description, and args_schema from the wrapped tool.
        
        Returns:
            ToolWithInstruction: The validated instance with updated attributes.
        """
        self.name = self.tool.name
        self.description = f"{self.tool.description}\nInstructions: {self.instructions}"
        self.args_schema = self.tool.args_schema
        return self
    
    def update_instructions(self, new_instructions: str) -> None:
        """Updates the tool's usage instructions.
        
        Args:
            new_instructions (str): New instructions for tool usage.
            
        Raises:
            ValueError: If new instructions are empty or exceed maximum length
        """
        if not new_instructions or not new_instructions.strip():
            raise ValueError("Instructions cannot be empty")
        
        if len(new_instructions) > self.MAX_INSTRUCTION_LENGTH:
            raise ValueError(
                f"Instructions exceed maximum length of {self.MAX_INSTRUCTION_LENGTH} characters"
            )
        
        self.instructions = new_instructions.strip()
        
        self.description = f"{self.tool.description}\nInstructions: {self.instructions}"
    
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped tool.
        
        Args:
            *args: Positional arguments to pass to the wrapped tool
            **kwargs: Keyword arguments to pass to the wrapped tool
            
        Returns:
            Any: The result from the wrapped tool's _run method
        """
        return self.tool._run(*args, **kwargs)
    
    def to_structured_tool(self) -> CrewStructuredTool:
        """Convert this tool to a CrewStructuredTool instance.
        
        Returns:
            CrewStructuredTool: A structured tool with instructions included in the description
        """
        structured_tool = self.tool.to_structured_tool()
        
        structured_tool.description = f"{structured_tool.description}\nInstructions: {self.instructions}"
        
        return structured_tool

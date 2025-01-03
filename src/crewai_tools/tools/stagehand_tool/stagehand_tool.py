"""
A tool for using Stagehand's AI-powered web automation capabilities in CrewAI.

This tool provides access to Stagehand's three core APIs:
- act: Perform web interactions
- extract: Extract information from web pages
- observe: Monitor web page changes

Each function takes atomic instructions to increase reliability.
"""

import os
from typing import Any, Type

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

# Define STAGEHAND_AVAILABLE at module level
STAGEHAND_AVAILABLE = False
try:
    import stagehand
    STAGEHAND_AVAILABLE = True
except ImportError:
    pass  # Keep STAGEHAND_AVAILABLE as False


class StagehandToolSchema(BaseModel):
    """Schema for the StagehandTool input parameters.
    
    Examples:
        ```python
        # Using the 'act' API to click a button
        tool.run(
            api_method="act",
            instruction="Click the 'Sign In' button"
        )
        
        # Using the 'extract' API to get text
        tool.run(
            api_method="extract",
            instruction="Get the text content of the main article"
        )
        
        # Using the 'observe' API to monitor changes
        tool.run(
            api_method="observe",
            instruction="Watch for changes in the shopping cart count"
        )
        ```
    """
    api_method: str = Field(
        ...,
        description="The Stagehand API to use: 'act' for interactions, 'extract' for getting content, or 'observe' for monitoring changes",
        pattern="^(act|extract|observe)$"
    )
    instruction: str = Field(
        ...,
        description="An atomic instruction for Stagehand to execute. Instructions should be simple and specific to increase reliability."
    )


class StagehandTool(BaseTool):
    """A tool for using Stagehand's AI-powered web automation capabilities.
    
    This tool provides access to Stagehand's three core APIs:
    - act: Perform web interactions (e.g., clicking buttons, filling forms)
    - extract: Extract information from web pages (e.g., getting text content)
    - observe: Monitor web page changes (e.g., watching for updates)
    
    Each function takes atomic instructions to increase reliability.
    
    Required Environment Variables:
        OPENAI_API_KEY: API key for OpenAI (required by Stagehand)
    
    Examples:
        ```python
        tool = StagehandTool()
        
        # Perform a web interaction
        result = tool.run(
            api_method="act",
            instruction="Click the 'Sign In' button"
        )
        
        # Extract content from a page
        content = tool.run(
            api_method="extract",
            instruction="Get the text content of the main article"
        )
        
        # Monitor for changes
        changes = tool.run(
            api_method="observe",
            instruction="Watch for changes in the shopping cart count"
        )
        ```
    """
    
    name: str = "StagehandTool"
    description: str = (
        "A tool that uses Stagehand's AI-powered web automation to interact with websites. "
        "It can perform actions (click, type, etc.), extract content, and observe changes. "
        "Each instruction should be atomic (simple and specific) to increase reliability."
    )
    args_schema: Type[BaseModel] = StagehandToolSchema
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the StagehandTool.
        
        The tool requires the OPENAI_API_KEY environment variable to be set.
        """
        super().__init__(**kwargs)
        
        if not STAGEHAND_AVAILABLE:
            raise ImportError(
                "The 'stagehand' package is required to use this tool. "
                "Please install it with: pip install stagehand"
            )
            
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for StagehandTool"
            )
    
    def _run(self, api_method: str, instruction: str, **kwargs: Any) -> Any:
        """Execute a Stagehand command using the specified API method.
        
        Args:
            api_method: The Stagehand API to use ('act', 'extract', or 'observe')
            instruction: An atomic instruction for Stagehand to execute
            **kwargs: Additional keyword arguments passed to the Stagehand API
            
        Returns:
            The result from the Stagehand API call
            
        Raises:
            ValueError: If an invalid api_method is provided
            RuntimeError: If the Stagehand API call fails
        """
        try:
            # Initialize Stagehand with the OpenAI API key
            st = stagehand.Stagehand(api_key=self.api_key)
            
            # Call the appropriate Stagehand API based on the method
            if api_method == "act":
                return st.act(instruction)
            elif api_method == "extract":
                return st.extract(instruction)
            elif api_method == "observe":
                return st.observe(instruction)
            else:
                raise ValueError(f"Unknown api_method: {api_method}")
                
        except Exception as e:
            raise RuntimeError(f"Stagehand API call failed: {str(e)}")

"""Tool for handling human input using LangGraph's interrupt mechanism."""

from typing import Any, Dict
from pydantic import Field

from crewai.tools import BaseTool

class HumanTool(BaseTool):
    """Tool for getting human input using LangGraph's interrupt mechanism."""
    
    name: str = "human"
    description: str = "Useful to ask user to enter input."
    result_as_answer: bool = False  # Don't use the response as final answer

    def _run(self, query: str) -> str:
        """Execute the human input tool.
        
        Args:
            query: The question to ask the user
            
        Returns:
            The user's response
            
        Raises:
            ImportError: If LangGraph is not installed
        """
        try:
            from langgraph.prebuilt.state_graphs import interrupt
            human_response = interrupt({"query": query})
            return human_response["data"]
        except ImportError:
            raise ImportError(
                "LangGraph is required for HumanTool. "
                "Install with `pip install langgraph`"
            )

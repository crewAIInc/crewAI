"""Tool for handling human input using LangGraph's interrupt mechanism."""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from crewai.tools import BaseTool


class HumanToolSchema(BaseModel):
    """Schema for HumanTool input validation."""
    query: str = Field(
        ...,
        description="The question to ask the user. Must be a non-empty string."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Optional timeout in seconds for waiting for user response"
    )

class HumanTool(BaseTool):
    """Tool for getting human input using LangGraph's interrupt mechanism.
    
    This tool allows agents to request input from users through LangGraph's
    interrupt mechanism. It supports timeout configuration and input validation.
    """
    
    name: str = "human"
    description: str = "Useful to ask user to enter input."
    args_schema: type[BaseModel] = HumanToolSchema
    result_as_answer: bool = False  # Don't use the response as final answer

    def _run(self, query: str, timeout: Optional[float] = None) -> str:
        """Execute the human input tool.
        
        Args:
            query: The question to ask the user
            timeout: Optional timeout in seconds
            
        Returns:
            The user's response
            
        Raises:
            ImportError: If LangGraph is not installed
            TimeoutError: If response times out
            ValueError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        try:
            from langgraph.prebuilt.state_graphs import interrupt
            logging.info(f"Requesting human input: {query}")
            human_response = interrupt({"query": query, "timeout": timeout})
            return human_response["data"]
        except ImportError:
            logging.error("LangGraph not installed")
            raise ImportError(
                "LangGraph is required for HumanTool. "
                "Install with `pip install langgraph`"
            )
        except Exception as e:
            logging.error(f"Error during human input: {str(e)}")
            raise

    async def _arun(self, query: str, timeout: Optional[float] = None) -> str:
        """Execute the human input tool asynchronously.
        
        Args:
            query: The question to ask the user
            timeout: Optional timeout in seconds
            
        Returns:
            The user's response
            
        Raises:
            ImportError: If LangGraph is not installed
            TimeoutError: If response times out
            ValueError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        try:
            from langgraph.prebuilt.state_graphs import interrupt
            logging.info(f"Requesting async human input: {query}")
            human_response = interrupt({"query": query, "timeout": timeout})
            return human_response["data"]
        except ImportError:
            logging.error("LangGraph not installed")
            raise ImportError(
                "LangGraph is required for HumanTool. "
                "Install with `pip install langgraph`"
            )
        except Exception as e:
            logging.error(f"Error during async human input: {str(e)}")
            raise

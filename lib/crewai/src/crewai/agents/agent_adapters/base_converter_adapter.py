"""Base converter adapter for structured output conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
    from crewai.task import Task


class BaseConverterAdapter(ABC):
    """Abstract base class for converter adapters in CrewAI.

    Defines the common interface for converting agent outputs to structured formats.
    All converter adapters must implement the methods defined here.
    """

    def __init__(self, agent_adapter: BaseAgentAdapter) -> None:
        """Initialize the converter adapter.

        Args:
            agent_adapter: The agent adapter to configure for structured output.
        """
        self.agent_adapter = agent_adapter

    @abstractmethod
    def configure_structured_output(self, task: Task) -> None:
        """Configure agents to return structured output.

        Must support both JSON and Pydantic output formats.

        Args:
            task: The task requiring structured output.
        """

    @abstractmethod
    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Enhance the system prompt with structured output instructions.

        Args:
            base_prompt: The original system prompt.

        Returns:
            Enhanced prompt with structured output guidance.
        """

    @abstractmethod
    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure proper string format.

        Args:
            result: The raw result from agent execution.

        Returns:
            Processed result as a string.
        """

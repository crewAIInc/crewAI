from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
    from crewai.task import Task


class BaseConverterAdapter(ABC):
    """Base class for all converter adapters in CrewAI.

    This abstract class defines the common interface and functionality that all
    converter adapters must implement for converting structured output.
    """

    def __init__(self, agent_adapter: BaseAgentAdapter) -> None:
        self.agent_adapter = agent_adapter

    @abstractmethod
    def configure_structured_output(self, task: Task) -> None:
        """Configure agents to return structured output.
        Must support json and pydantic output.
        """

    @abstractmethod
    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Enhance the system prompt with structured output instructions."""

    @abstractmethod
    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure it matches the expected format: string."""

"""Base converter adapter for structured output conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import TYPE_CHECKING, Any, Final, Literal

from crewai.utilities.pydantic_schema_utils import generate_model_description


if TYPE_CHECKING:
    from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
    from crewai.task import Task


_CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```"
)
_JSON_OBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{[\s\S]*}")


class BaseConverterAdapter(ABC):
    """Abstract base class for converter adapters in CrewAI.

    Defines the common interface for converting agent outputs to structured formats.
    All converter adapters must implement the methods defined here.

    Attributes:
        agent_adapter: The agent adapter instance.
        _output_format: The expected output format (json, pydantic, or None).
        _schema: The schema description for the expected output.
    """

    def __init__(self, agent_adapter: BaseAgentAdapter) -> None:
        """Initialize the converter adapter.

        Args:
            agent_adapter: The agent adapter to configure for structured output.
        """
        self.agent_adapter = agent_adapter
        self._output_format: Literal["json", "pydantic"] | None = None
        self._schema: dict[str, Any] | None = None

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

    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure proper string format.

        Extracts valid JSON from text that may contain markdown or other formatting.

        Args:
            result: The raw result from agent execution.

        Returns:
            Processed result as a string.
        """
        if not self._output_format:
            return result

        return self._extract_json_from_text(result)

    @staticmethod
    def _validate_json(text: str) -> str | None:
        """Validate if text is valid JSON and return it, or None if invalid.

        Args:
            text: The text to validate as JSON.

        Returns:
            The text if it's valid JSON, None otherwise.
        """
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_json_from_text(result: str) -> str:
        """Extract valid JSON from text that may contain markdown or other formatting.

        This method provides a comprehensive approach to extracting JSON from LLM responses,
        handling cases where JSON may be wrapped in Markdown code blocks or embedded in text.

        Args:
            result: The text potentially containing JSON.

        Returns:
            Extracted JSON string if found and valid, otherwise the original result.
        """
        if not isinstance(result, str):
            return str(result)

        if valid := BaseConverterAdapter._validate_json(result):
            return valid

        for match in _CODE_BLOCK_PATTERN.finditer(result):
            if valid := BaseConverterAdapter._validate_json(match.group(1).strip()):
                return valid

        for match in _JSON_OBJECT_PATTERN.finditer(result):
            if valid := BaseConverterAdapter._validate_json(match.group()):
                return valid

        return result

    @staticmethod
    def _configure_format_from_task(
        task: Task,
    ) -> tuple[Literal["json", "pydantic"] | None, dict[str, Any] | None]:
        """Determine output format and schema from task requirements.

        This is a helper method that examines the task's output requirements
        and returns the appropriate format type and schema description.

        Args:
            task: The task containing output format requirements.

        Returns:
            A tuple of (output_format, schema) where both may be None if no
            structured output is required.
        """

        if not (task.output_json or task.output_pydantic):
            return None, None

        if task.output_json:
            return "json", generate_model_description(task.output_json)
        if task.output_pydantic:
            return "pydantic", generate_model_description(task.output_pydantic)

        return None, None

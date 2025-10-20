"""LangGraph structured output converter for CrewAI task integration.

This module contains the LangGraphConverterAdapter class that handles structured
output conversion for LangGraph agents, supporting JSON and Pydantic model formats.
"""

from typing import Any

from crewai.agents.agent_adapters.base_converter_adapter import BaseConverterAdapter


class LangGraphConverterAdapter(BaseConverterAdapter):
    """Adapter for handling structured output conversion in LangGraph agents.

    Converts task output requirements into system prompt modifications and
    post-processing logic to ensure agents return properly structured outputs.

    Attributes:
        _system_prompt_appendix: Cached system prompt instructions for structured output.
    """

    def __init__(self, agent_adapter: Any) -> None:
        """Initialize the converter adapter with a reference to the agent adapter.

        Args:
            agent_adapter: The LangGraph agent adapter instance.
        """
        super().__init__(agent_adapter=agent_adapter)
        self.agent_adapter: Any = agent_adapter
        self._system_prompt_appendix: str | None = None

    def configure_structured_output(self, task: Any) -> None:
        """Configure the structured output for LangGraph.

        Analyzes the task's output requirements and sets up the necessary
        formatting and validation logic.

        Args:
            task: The task object containing output format specifications.
        """
        self._output_format, self._schema = self._configure_format_from_task(task)
        self._system_prompt_appendix = self._generate_system_prompt_appendix()

    def _generate_system_prompt_appendix(self) -> str:
        """Generate an appendix for the system prompt to enforce structured output.

        Creates instructions that are appended to the system prompt to guide
        the agent in producing properly formatted output.

        Returns:
            System prompt appendix string, or empty string if no structured output.
        """
        if not self._output_format or not self._schema:
            return ""

        return f"""
Important: Your final answer MUST be provided in the following structured format:

{self._schema}

DO NOT include any markdown code blocks, backticks, or other formatting around your response.
The output should be raw JSON that exactly matches the specified schema.
"""

    def enhance_system_prompt(self, original_prompt: str) -> str:
        """Add structured output instructions to the system prompt if needed.

        Args:
            original_prompt: The base system prompt.

        Returns:
            Enhanced system prompt with structured output instructions.
        """
        if not self._system_prompt_appendix:
            return original_prompt

        return f"{original_prompt}\n{self._system_prompt_appendix}"

"""OpenAI structured output converter for CrewAI task integration.

This module contains the OpenAIConverterAdapter class that handles structured
output conversion for OpenAI agents, supporting JSON and Pydantic model formats.
"""

import json
from typing import Any

from crewai.agents.agent_adapters.base_converter_adapter import BaseConverterAdapter
from crewai.utilities.i18n import get_i18n


class OpenAIConverterAdapter(BaseConverterAdapter):
    """Adapter for handling structured output conversion in OpenAI agents.

    This adapter enhances the OpenAI agent to handle structured output formats
    and post-processes the results when needed.

    Attributes:
        _output_model: The Pydantic model for the output (OpenAI-specific).
    """

    def __init__(self, agent_adapter: Any) -> None:
        """Initialize the converter adapter with a reference to the agent adapter.

        Args:
            agent_adapter: The OpenAI agent adapter instance.
        """
        super().__init__(agent_adapter=agent_adapter)
        self.agent_adapter: Any = agent_adapter
        self._output_model: Any = None

    def configure_structured_output(self, task: Any) -> None:
        """Configure the structured output for OpenAI agent based on task requirements.

        Args:
            task: The task containing output format requirements.
        """
        self._output_format, self._schema = self._configure_format_from_task(task)
        self._output_model = None

        if task.output_json:
            self.agent_adapter._openai_agent.output_type = task.output_json
            self._output_model = task.output_json
        elif task.output_pydantic:
            self.agent_adapter._openai_agent.output_type = task.output_pydantic
            self._output_model = task.output_pydantic

    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Enhance the base system prompt with structured output requirements if needed.

        Args:
            base_prompt: The original system prompt.

        Returns:
            Enhanced system prompt with output format instructions if needed.
        """
        if not self._output_format:
            return base_prompt

        output_schema: str = (
            get_i18n()
            .slice("formatted_task_instructions")
            .format(output_format=json.dumps(self._schema, indent=2))
        )

        return f"{base_prompt}\n\n{output_schema}"

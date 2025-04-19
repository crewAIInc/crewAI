import json
import re

from crewai.agents.agent_adapters.base_converter_adapter import BaseConverterAdapter
from crewai.utilities.converter import generate_model_description
from crewai.utilities.i18n import I18N


class OpenAIConverterAdapter(BaseConverterAdapter):
    """
    Adapter for handling structured output conversion in OpenAI agents.

    This adapter enhances the OpenAI agent to handle structured output formats
    and post-processes the results when needed.

    Attributes:
        _output_format: The expected output format (json, pydantic, or None)
        _schema: The schema description for the expected output
        _output_model: The Pydantic model for the output
    """

    def __init__(self, agent_adapter):
        """Initialize the converter adapter with a reference to the agent adapter"""
        self.agent_adapter = agent_adapter
        self._output_format = None
        self._schema = None
        self._output_model = None

    def configure_structured_output(self, task) -> None:
        """
        Configure the structured output for OpenAI agent based on task requirements.

        Args:
            task: The task containing output format requirements
        """
        # Reset configuration
        self._output_format = None
        self._schema = None
        self._output_model = None

        # If no structured output is required, return early
        if not (task.output_json or task.output_pydantic):
            return

        # Configure based on task output format
        if task.output_json:
            self._output_format = "json"
            self._schema = generate_model_description(task.output_json)
            self.agent_adapter._openai_agent.output_type = task.output_json
            self._output_model = task.output_json
        elif task.output_pydantic:
            self._output_format = "pydantic"
            self._schema = generate_model_description(task.output_pydantic)
            self.agent_adapter._openai_agent.output_type = task.output_pydantic
            self._output_model = task.output_pydantic

    def enhance_system_prompt(self, base_prompt: str) -> str:
        """
        Enhance the base system prompt with structured output requirements if needed.

        Args:
            base_prompt: The original system prompt

        Returns:
            Enhanced system prompt with output format instructions if needed
        """
        if not self._output_format:
            return base_prompt

        output_schema = (
            I18N()
            .slice("formatted_task_instructions")
            .format(output_format=self._schema)
        )

        return f"{base_prompt}\n\n{output_schema}"

    def post_process_result(self, result: str) -> str:
        """
        Post-process the result to ensure it matches the expected format.

        This method attempts to extract valid JSON from the result if necessary.

        Args:
            result: The raw result from the agent

        Returns:
            Processed result conforming to the expected output format
        """
        if not self._output_format:
            return result
        # Try to extract valid JSON if it's wrapped in code blocks or other text
        if isinstance(result, str) and self._output_format in ["json", "pydantic"]:
            # First, try to parse as is
            try:
                json.loads(result)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(code_block_pattern, result)

                for block in code_blocks:
                    try:
                        json.loads(block.strip())
                        return block.strip()
                    except json.JSONDecodeError:
                        continue

                # Try to extract any JSON-like structure
                json_pattern = r"(\{[\s\S]*\})"
                json_matches = re.findall(json_pattern, result, re.DOTALL)

                for match in json_matches:
                    try:
                        json.loads(match)
                        return match
                    except json.JSONDecodeError:
                        continue

        # If all extraction attempts fail, return the original
        return str(result)

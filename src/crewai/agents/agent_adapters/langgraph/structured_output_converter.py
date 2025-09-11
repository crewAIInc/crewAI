"""LangGraph structured output converter for CrewAI task integration.

This module contains the LangGraphConverterAdapter class that handles structured
output conversion for LangGraph agents, supporting JSON and Pydantic model formats.
"""

import json
import re
from typing import Any, Literal

from crewai.agents.agent_adapters.base_converter_adapter import BaseConverterAdapter
from crewai.utilities.converter import generate_model_description


class LangGraphConverterAdapter(BaseConverterAdapter):
    """Adapter for handling structured output conversion in LangGraph agents.

    Converts task output requirements into system prompt modifications and
    post-processing logic to ensure agents return properly structured outputs.
    """

    def __init__(self, agent_adapter: Any) -> None:
        """Initialize the converter adapter with a reference to the agent adapter.

        Args:
            agent_adapter: The LangGraph agent adapter instance.
        """
        super().__init__(agent_adapter=agent_adapter)
        self.agent_adapter: Any = agent_adapter
        self._output_format: Literal["json", "pydantic"] | None = None
        self._schema: str | None = None
        self._system_prompt_appendix: str | None = None

    def configure_structured_output(self, task: Any) -> None:
        """Configure the structured output for LangGraph.

        Analyzes the task's output requirements and sets up the necessary
        formatting and validation logic.

        Args:
            task: The task object containing output format specifications.
        """
        if not (task.output_json or task.output_pydantic):
            self._output_format = None
            self._schema = None
            self._system_prompt_appendix = None
            return

        if task.output_json:
            self._output_format = "json"
            self._schema = generate_model_description(task.output_json)
        elif task.output_pydantic:
            self._output_format = "pydantic"
            self._schema = generate_model_description(task.output_pydantic)

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

    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure it matches the expected format.

        Attempts to extract and validate JSON content from agent responses,
        handling cases where JSON may be wrapped in markdown or other formatting.

        Args:
            result: The raw result string from the agent.

        Returns:
            Processed result string, ideally in valid JSON format.
        """
        if not self._output_format:
            return result

        # Try to extract valid JSON if it's wrapped in code blocks or other text
        if self._output_format in ["json", "pydantic"]:
            try:
                # First, try to parse as is
                json.loads(result)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                json_match: re.Match[str] | None = re.search(
                    r"(\{.*})", result, re.DOTALL
                )
                if json_match:
                    try:
                        extracted: str = json_match.group(1)
                        # Validate it's proper JSON
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        pass

        return result

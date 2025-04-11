import json

from crewai.agents.agent_adapters.base_converter_adapter import BaseConverterAdapter
from crewai.utilities.converter import generate_model_description


class LangGraphConverterAdapter(BaseConverterAdapter):
    """Adapter for handling structured output conversion in LangGraph agents"""

    def __init__(self, agent_adapter):
        """Initialize the converter adapter with a reference to the agent adapter"""
        self.agent_adapter = agent_adapter
        self._output_format = None
        self._schema = None
        self._system_prompt_appendix = None

    def configure_structured_output(self, task) -> None:
        """Configure the structured output for LangGraph."""
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
        """Generate an appendix for the system prompt to enforce structured output"""
        if not self._output_format or not self._schema:
            return ""

        return f"""
Important: Your final answer MUST be provided in the following structured format:

{self._schema}

DO NOT include any markdown code blocks, backticks, or other formatting around your response. 
The output should be raw JSON that exactly matches the specified schema.
"""

    def enhance_system_prompt(self, original_prompt: str) -> str:
        """Add structured output instructions to the system prompt if needed"""
        if not self._system_prompt_appendix:
            return original_prompt

        return f"{original_prompt}\n{self._system_prompt_appendix}"

    def post_process_result(self, result: str) -> str:
        """Post-process the result to ensure it matches the expected format"""
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
                import re

                json_match = re.search(r"(\{.*\})", result, re.DOTALL)
                if json_match:
                    try:
                        extracted = json_match.group(1)
                        # Validate it's proper JSON
                        json.loads(extracted)
                        return extracted
                    except:
                        pass

        return result

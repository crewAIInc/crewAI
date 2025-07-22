import json
import re
from typing import Any, get_origin

from pydantic import BaseModel, ValidationError

from crewai.utilities.converter import Converter, ConverterError


class TrainingConverter(Converter):
    """
    A specialized converter for smaller LLMs (up to 7B parameters) that handles validation errors
    by breaking down the model into individual fields and querying the LLM for each field separately.
    """

    def to_pydantic(self, current_attempt=1) -> BaseModel:
        try:
            return super().to_pydantic(current_attempt)
        except ConverterError:
            return self._convert_field_by_field()

    def _convert_field_by_field(self) -> BaseModel:
        field_values = {}

        for field_name, field_info in self.model.model_fields.items():
            field_description = field_info.description
            field_type = field_info.annotation

            response = self._ask_llm_for_field(field_name, field_description)
            value = self._process_field_value(response, field_type)
            field_values[field_name] = value

        try:
            return self.model(**field_values)
        except ValidationError as e:
            raise ConverterError(f"Failed to create model from individually collected fields: {e}")

    def _ask_llm_for_field(self, field_name: str, field_description: str) -> str:
        prompt = f"""
Based on the following information:
{self.text}

Please provide ONLY the {field_name} field value as described:
"{field_description}"

Respond with ONLY the requested information, nothing else.
"""
        return self.llm.call([
            {"role": "system", "content": f"Extract the {field_name} from the previous information."},
            {"role": "user", "content": prompt}
        ])

    def _process_field_value(self, response: str, field_type: Any) -> Any:
        response = response.strip()
        origin = get_origin(field_type)

        if origin is list:
            return self._parse_list(response)

        if field_type is float:
            return self._parse_float(response)

        if field_type is str:
            return response

        return response

    def _parse_list(self, response: str) -> list:
        try:
            if response.startswith('['):
                return json.loads(response)

            items = [item.strip() for item in response.split('\n') if item.strip()]
            return [self._strip_bullet(item) for item in items]

        except json.JSONDecodeError:
            return [response]

    def _parse_float(self, response: str) -> float:
        try:
            match = re.search(r'(\d+(\.\d+)?)', response)
            return float(match.group(1)) if match else 0.0
        except Exception:
            return 0.0

    def _strip_bullet(self, item: str) -> str:
        if item.startswith(('- ', '* ')):
            return item[2:].strip()
        return item.strip()
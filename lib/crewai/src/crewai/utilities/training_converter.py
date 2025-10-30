import json
import re
from typing import Any, Final, get_origin

from pydantic import BaseModel, ValidationError

from crewai.utilities.converter import Converter, ConverterError


_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\d+(?:\.\d+)?)")


class TrainingConverter(Converter):
    """A specialized converter for smaller LLMs (up to 7B parameters) that handles validation errors
    by breaking down the model into individual fields and querying the LLM for each field separately.
    """

    def to_pydantic(self, current_attempt: int = 1) -> BaseModel:
        """Convert the text to a Pydantic model, with fallback to field-by-field extraction on failure.

        Args:
            current_attempt: The current attempt number for conversion.

        Returns:
            An instance of the Pydantic model.

        Raises:
            ConverterError: If conversion fails after field-by-field extraction.
        """
        try:
            return super().to_pydantic(current_attempt)
        except ConverterError:
            return self._convert_field_by_field()

    def _convert_field_by_field(self) -> BaseModel:
        field_values: dict[str, Any] = {}

        for field_name, field_info in self.model.model_fields.items():
            field_description: str | None = field_info.description
            field_type: type | None = field_info.annotation

            if field_description is None:
                raise ValueError(f"Field '{field_name}' has no description")
            response: str = self._ask_llm_for_field(
                field_name=field_name, field_description=field_description
            )
            value: Any = self._process_field_value(
                response=response, field_type=field_type
            )
            field_values[field_name] = value

        try:
            return self.model(**field_values)
        except ValidationError as e:
            raise ConverterError(
                f"Failed to create model from individually collected fields: {e}"
            ) from e

    def _ask_llm_for_field(self, field_name: str, field_description: str) -> str:
        """Query the LLM for a specific field value based on its description.

        Args:
            field_name: The name of the field to extract.
            field_description: The description of the field to guide extraction.

        Returns:
            The LLM's response containing the field value.
        """

        prompt: str = f"""
Based on the following information:
{self.text}

Please provide ONLY the {field_name} field value as described:
"{field_description}"

Respond with ONLY the requested information, nothing else.
"""
        return self.llm.call(
            [
                {
                    "role": "system",
                    "content": f"Extract the {field_name} from the previous information.",
                },
                {"role": "user", "content": prompt},
            ]
        )

    def _process_field_value(self, response: str, field_type: type | None) -> Any:
        response = response.strip()
        origin: type[Any] | None = get_origin(field_type)

        if origin is list:
            return self._parse_list(response)

        if field_type is float:
            return self._parse_float(response)

        if field_type is str:
            return response

        return response

    def _parse_list(self, response: str) -> list[Any]:
        try:
            if response.startswith("["):
                return json.loads(response)

            items: list[str] = [
                item.strip() for item in response.split("\n") if item.strip()
            ]
            return [self._strip_bullet(item) for item in items]

        except json.JSONDecodeError:
            return [response]

    @staticmethod
    def _parse_float(response: str) -> float:
        """Parse a float from the response, extracting the first numeric value found.

        Args:
            response: The response string from which to extract the float.

        Returns:
            The extracted float value, or 0.0 if no valid float is found.
        """
        try:
            match = _FLOAT_PATTERN.search(response)
            return float(match.group(1)) if match else 0.0
        except (ValueError, AttributeError):
            return 0.0

    @staticmethod
    def _strip_bullet(item: str) -> str:
        """Remove common bullet point characters from the start of a string.

        Args:
            item: The string item to process.

        Returns:
            The string without leading bullet characters.
        """
        if item.startswith(("- ", "* ")):
            return item[2:].strip()
        return item.strip()

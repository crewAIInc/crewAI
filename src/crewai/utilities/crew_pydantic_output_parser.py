import json
from typing import Any, Type

import regex
from pydantic import BaseModel, ValidationError

from crewai.agents.parser import OutputParserException

"""Parser for converting text outputs into Pydantic models."""


def clean_json_from_text(text: str) -> str:
    """Extract and clean JSON from text that may contain markdown or trailing characters."""
    text = text.replace("```", "").replace("json", "")
    json_pattern = r"\{(?:[^{}]|(?R))*\}"
    matches = regex.finditer(json_pattern, text)

    for match in matches:
        try:
            json_obj = json.loads(match.group())
            json_obj = json.dumps(json_obj)
            return str(json_obj)
        except json.JSONDecodeError:
            continue
    return text

class CrewPydanticOutputParser:
    """Parses text outputs into specified Pydantic models."""

    pydantic_object: Type[BaseModel]

    def parse_result(self, result: str) -> Any:
        result = self._transform_in_valid_json(result)

        # Treating edge case of function calling llm returning the name instead of tool_name
        json_object = json.loads(result)
        if "tool_name" not in json_object:
            json_object["tool_name"] = json_object.get("name", "")
        result = json.dumps(json_object)

        try:
            return self.pydantic_object.model_validate(json_object)
        except ValidationError as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(error=msg)

    def _transform_in_valid_json(self, text) -> str:
        return clean_json_from_text(text)

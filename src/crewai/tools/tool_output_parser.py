import json
from typing import Any, List

import regex
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import ValidationError


class ToolOutputParser(PydanticOutputParser):
    """Parses the function calling of a tool usage and it's arguments."""

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        result[0].text = self._transform_in_valid_json(result[0].text)
        json_object = super().parse_result(result)
        try:
            return self.pydantic_object.parse_obj(json_object)
        except ValidationError as e:
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)

    def _transform_in_valid_json(self, text) -> str:
        text = text.replace("```", "").replace("json", "")
        json_pattern = r"\{(?:[^{}]|(?R))*\}"
        matches = regex.finditer(json_pattern, text)

        for match in matches:
            try:
                # Attempt to parse the matched string as JSON
                json_obj = json.loads(match.group())
                # Return the first successfully parsed JSON object
                json_obj = json.dumps(json_obj)
                return str(json_obj)
            except json.JSONDecodeError:
                # If parsing fails, skip to the next match
                continue
        return text

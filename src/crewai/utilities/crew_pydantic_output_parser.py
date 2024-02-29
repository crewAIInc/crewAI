import json
from typing import Any, List, Type, Union

import regex
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from langchain_core.pydantic_v1 import ValidationError
from pydantic import BaseModel
from pydantic.v1 import BaseModel as V1BaseModel


class CrewPydanticOutputParser(PydanticOutputParser):
    """This class is responsible for parsing the generated text into Pydantic models."""

    pydantic_object: Union[Type[BaseModel], Type[V1BaseModel]]  # The Pydantic model to parse the text into

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        """This method parses the result of the language model generation into a Pydantic object."""

        # Transform the first result into valid JSON
        result[0].text = self._transform_in_valid_json(result[0].text)

        # Parse the result using the parent class method
        json_object = super().parse_result(result)

        try:
            # Try to parse the JSON object into the Pydantic model
            return self.pydantic_object.parse_obj(json_object)
        except ValidationError as e:
            # If parsing fails, raise an exception with a detailed error message
            name = self.pydantic_object.__name__
            msg = f"Failed to parse {name} from completion {json_object}. Got: {e}"
            raise OutputParserException(msg, llm_output=json_object)

    def _transform_in_valid_json(self, text) -> str:
        """This method transforms the given text into a valid JSON string."""

        # Remove unnecessary characters that might interfere with JSON parsing
        text = text.replace("```", "").replace("json", "")

        # Define a regex pattern to match JSON objects
        json_pattern = r"\{(?:[^{}]|(?R))*\}"
        matches = regex.finditer(json_pattern, text)

        for match in matches:
            try:
                # Attempt to parse the matched string as JSON
                json_obj = json.loads(match.group())

                # If parsing is successful, return the JSON object as a string
                json_obj = json.dumps(json_obj)
                return str(json_obj)
            except json.JSONDecodeError:
                # If parsing fails, skip to the next match
                continue

        # If no valid JSON object is found, return the original text
        return text

import json

from pydantic import model_validator

from crewai.agents.third_party_agents.utilities.converter_base import AbstractConverter
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai import OpenAI


class ConverterError(Exception):
    """Error raised when Converter fails to parse the input."""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message


class LLamaOutputConverter(AbstractConverter):
    """Class that converts text into either pydantic or json."""

    @model_validator(mode="after")
    def check_llm_provider(self):
        if not self._is_gpt(self.llm):
            self._is_gpt = False

    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        try:
            if self._is_gpt:
                return self._get_llm_response()
            else:
                return self._create_runnable_step().chat()
        except Exception as e:
            if current_attempt < self.max_attemps:
                return self.to_pydantic(current_attempt + 1)
            return ConverterError(
                f"Failed to convert text into a pydantic model due to the following error: {e}"
            )

    def to_json(self, current_attempt=1):
        """Convert text to json."""
        try:
            if self._is_gpt:
                return self._get_llm_response()

            else:
                return json.dumps(self._create_runnable_step().model_dump())

        except Exception as e:
            print("error", e)
            if current_attempt < self.max_attemps:
                return self.to_json(current_attempt + 1)
            return ConverterError("Failed to convert text into JSON.")

    def _create_instructor(self):
        """Create an instructor."""
        from crewai.utilities import Instructor

        inst = Instructor(
            llm=self.llm,
            max_attemps=self.max_attemps,
            model=self.model,
            content=self.text,
            instructions=self.instructions,
        )
        return inst

    def _create_runnable_step(self):
        """Given an LLM as well as an output Pydantic class, generate a structured Pydantic object."""
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=self.model,
            prompt_template_str=self.text,
            verbose=True,
        )
        return program(raw_text=self.text)

    def _get_llm_response(self):
        self.llm.output_parser = LLMTextCompletionProgram.from_defaults(
            output_cls=self.model,
            prompt_template_str=self.instructions,
            verbose=True,
        )._prompt
        messages = [{"role": "user", "content": self.text}]
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        result = self.llm._client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
        )
        return result.choices[0].message.content

    def _is_gpt(self, llm) -> bool:  # type: ignore # BUG? Name "_is_gpt" defined on line 20 hides name from outer scope
        return isinstance(llm, OpenAI) and llm.api_base == "https://api.openai.com/v1"

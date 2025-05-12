from typing import Any

import aisuite as ai

from crewai.llms.base_llm import BaseLLM


class AISuiteLLM(BaseLLM):
    def __init__(self, model: str, temperature: float | None = None, **kwargs) -> None:
        super().__init__(model, temperature, **kwargs)
        self.client = ai.Client()

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
    ) -> str | Any:
        completion_params = self._prepare_completion_params(messages, tools)
        response = self.client.chat.completions.create(**completion_params)

        return response.choices[0].message.content

    def _prepare_completion_params(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
        }

    def supports_function_calling(self) -> bool:
        return False

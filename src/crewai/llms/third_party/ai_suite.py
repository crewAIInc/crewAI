from typing import Any, Optional, Union

import aisuite as ai

from crewai.llms.base_llm import BaseLLM


class AISuiteLLM(BaseLLM):
    def __init__(self, model: str, temperature: Optional[float] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.client = ai.Client()

    def call(
        self,
        messages: Union[str, list[dict[str, str]]],
        tools: Optional[list[dict]] = None,
        callbacks: Optional[list[Any]] = None,
        available_functions: Optional[dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> Union[str, Any]:
        completion_params = self._prepare_completion_params(messages, tools)
        response = self.client.chat.completions.create(**completion_params)

        return response.choices[0].message.content

    def _prepare_completion_params(
        self,
        messages: Union[str, list[dict[str, str]]],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
        }

    def supports_function_calling(self) -> bool:
        return False

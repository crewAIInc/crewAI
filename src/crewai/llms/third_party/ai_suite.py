from typing import Any, Dict, List, Optional

import aisuite as ai

from crewai.llms.base_llm import BaseLLM


class AISuiteLLM(BaseLLM):
    def __init__(self, model: str, temperature: Optional[float] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.client = ai.Client()

    def call(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        completion_params = self._prepare_completion_params(messages)
        # print(f"Completion params: {completion_params}")
        response = self.client.chat.completions.create(**completion_params)
        print(f"Response: {response}")
        tool_calls = getattr(response.choices[0].message, "tool_calls", [])
        print(f"Tool calls: {tool_calls}")
        return response.choices[0].message.content

    def _prepare_completion_params(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        print(f"Preparing completion params for {self.model}")
        # print(f"Messages: {messages}")
        print(f"Temperature: {self.temperature}")
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self):
        pass

    def set_callbacks(self, callbacks: List[Any]) -> None:
        pass

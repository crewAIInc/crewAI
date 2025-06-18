from typing import Any, Dict, List, Optional, Union

try:
    import aisuite as ai
    AISUITE_AVAILABLE = True
except ImportError:
    AISUITE_AVAILABLE = False
    ai = None

from crewai.llms.base_llm import BaseLLM


class AISuiteLLM(BaseLLM):
    def __init__(self, model: str, temperature: Optional[float] = None, **kwargs):
        if not AISUITE_AVAILABLE:
            raise ImportError(
                "AISuite is required for AISuiteLLM. "
                "Please install it with: pip install 'crewai[llm-integrations]'"
            )
        super().__init__(model, temperature, **kwargs)
        self.client = ai.Client()

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        completion_params = self._prepare_completion_params(messages, tools)
        response = self.client.chat.completions.create(**completion_params)

        return response.choices[0].message.content

    def _prepare_completion_params(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
        }

    def supports_function_calling(self) -> bool:
        return False

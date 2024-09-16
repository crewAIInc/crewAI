from typing import Any, Dict, List
from litellm import completion
import litellm


class LLM:
    def __init__(self, model: str, stop: List[str] = [], callbacks: List[Any] = []):
        self.stop = stop
        self.model = model
        litellm.callbacks = callbacks

    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = completion(
            stop=self.stop, model=self.model, messages=messages, num_retries=5
        )
        return response["choices"][0]["message"]["content"]

    def _call_callbacks(self, formatted_answer):
        for callback in self.callbacks:
            callback(formatted_answer)

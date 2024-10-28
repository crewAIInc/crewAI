from contextlib import contextmanager, suppress
from typing import Any, Dict, List, Optional, Union
import logging
from functools import lru_cache
import litellm
from litellm import get_supported_openai_params

from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)

import sys
import io


class FilteredStream(io.StringIO):
    def write(self, s):
        if (
            "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
            in s
            or "LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True`"
            in s
        ):
            return
        super().write(s)


LLM_CONTEXT_WINDOW_SIZES = {
    # openai
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    # deepseek
    "deepseek-chat": 128000,
    # groq
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
}


@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Redirect stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = FilteredStream()
        sys.stderr = FilteredStream()

        try:
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class LLM:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self._setup_params(kwargs)
        self._configure_litellm()
    
    def _setup_params(self, kwargs: Dict[str, Any]) -> None:
        self.params = {
            k: v for k, v in kwargs.items() 
            if v is not None and k in get_supported_openai_params(self.model)
        }
        
    @lru_cache(maxsize=128)
    def call(self, messages: List[Dict[str, str]], callbacks: List[Any] = []) -> str:
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **self.params
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"LiteLLM call failed: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def get_context_window_size(self) -> int:
        return int(LLM_CONTEXT_WINDOW_SIZES.get(self.model, 8192) * 0.75)

from typing import Any, Dict, List

import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class TokenProcess:
    """This class is responsible for tracking the number of tokens and successful requests.
    It provides methods to increment these counts and to get a summary of the counts."""

    total_tokens: int = 0  # Total number of tokens
    prompt_tokens: int = 0  # Number of tokens in prompts
    completion_tokens: int = 0  # Number of tokens in completions
    successful_requests: int = 0  # Number of successful requests

    def sum_prompt_tokens(self, tokens: int):
        """Increments the count of prompt tokens and total tokens by the specified number."""
        self.prompt_tokens = self.prompt_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_completion_tokens(self, tokens: int):
        """Increments the count of completion tokens and total tokens by the specified number."""
        self.completion_tokens = self.completion_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_successful_requests(self, requests: int):
        """Increments the count of successful requests by the specified number."""
        self.successful_requests = self.successful_requests + requests

    def get_summary(self) -> str:
        """Returns a summary of the counts as a dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests,
        }


class TokenCalcHandler(BaseCallbackHandler):
    """This class is a callback handler for token calculation.
    It uses a TokenProcess to track the number of tokens and successful requests."""

    model: str = ""  # The model name
    token_cost_process: TokenProcess  # The TokenProcess to track the counts

    def __init__(self, model, token_cost_process):
        """Initializes the handler with the specified model and TokenProcess."""
        self.model = model
        self.token_cost_process = token_cost_process

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Handles the start of a language model request.
        It encodes the prompts and increments the count of prompt tokens by the number of tokens in the prompts."""
        if "gpt" in self.model:
            encoding = tiktoken.encoding_for_model(self.model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        if self.token_cost_process == None:
            return

        for prompt in prompts:
            self.token_cost_process.sum_prompt_tokens(len(encoding.encode(prompt)))

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handles the generation of a new token.
        It increments the count of completion tokens by 1."""
        self.token_cost_process.sum_completion_tokens(1)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handles the end of a language model request.
        It increments the count of successful requests by 1."""
        self.token_cost_process.sum_successful_requests(1)

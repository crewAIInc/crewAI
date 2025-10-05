"""Token counting callback handler for LLM interactions.

This module provides a callback handler that tracks token usage
for LLM API calls through the litellm library.
"""

from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.utilities.logger_utils import suppress_warnings


class TokenCalcHandler(CustomLogger):
    """Handler for calculating and tracking token usage in LLM calls.

    This handler integrates with litellm's logging system to track
    prompt tokens, completion tokens, and cached tokens across requests.

    Attributes:
        token_cost_process: The token process tracker to accumulate usage metrics.
    """

    def __init__(self, token_cost_process: TokenProcess | None, **kwargs: Any) -> None:
        """Initialize the token calculation handler.

        Args:
            token_cost_process: Optional token process tracker for accumulating metrics.
        """
        super().__init__(**kwargs)
        self.token_cost_process = token_cost_process

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """Log successful LLM API call and track token usage.

        Args:
            kwargs: The arguments passed to the LLM call.
            response_obj: The response object from the LLM API.
            start_time: The timestamp when the call started.
            end_time: The timestamp when the call completed.
        """
        if self.token_cost_process is None:
            return

        with suppress_warnings():
            if isinstance(response_obj, dict) and "usage" in response_obj:
                usage: Usage = response_obj["usage"]
                if usage:
                    self.token_cost_process.sum_successful_requests(1)
                    if hasattr(usage, "prompt_tokens"):
                        self.token_cost_process.sum_prompt_tokens(usage.prompt_tokens)
                    if hasattr(usage, "completion_tokens"):
                        self.token_cost_process.sum_completion_tokens(
                            usage.completion_tokens
                        )
                    if (
                        hasattr(usage, "prompt_tokens_details")
                        and usage.prompt_tokens_details
                        and usage.prompt_tokens_details.cached_tokens
                    ):
                        self.token_cost_process.sum_cached_prompt_tokens(
                            usage.prompt_tokens_details.cached_tokens
                        )

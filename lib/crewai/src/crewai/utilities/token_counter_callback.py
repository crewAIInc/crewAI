"""Token counting callback handler for LLM interactions.

This module provides a callback handler that tracks token usage
for LLM API calls. Works standalone and also integrates with litellm
when available (for the litellm fallback path).
"""

from typing import Any

from pydantic import BaseModel, Field

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.utilities.logger_utils import suppress_warnings


# Check if litellm is available for callback integration
try:
    from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger

    LITELLM_AVAILABLE = True
except ImportError:
    LiteLLMCustomLogger = None  # type: ignore[misc, assignment]
    LITELLM_AVAILABLE = False


class TokenCalcHandler(BaseModel):
    """Handler for calculating and tracking token usage in LLM calls.

    This handler tracks prompt tokens, completion tokens, and cached tokens
    across requests. It works standalone and also integrates with litellm's
    logging system when litellm is installed (for the fallback path).
    """

    model_config = {"arbitrary_types_allowed": True}

    __hash__ = object.__hash__

    token_cost_process: TokenProcess | None = Field(default=None)

    def __init__(
        self, token_cost_process: TokenProcess | None = None, /, **kwargs: Any
    ) -> None:
        if token_cost_process is not None:
            kwargs["token_cost_process"] = token_cost_process
        super().__init__(**kwargs)

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """Log successful LLM API call and track token usage."""
        if self.token_cost_process is None:
            return

        with suppress_warnings():
            if isinstance(response_obj, dict) and "usage" in response_obj:
                usage = response_obj["usage"]
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

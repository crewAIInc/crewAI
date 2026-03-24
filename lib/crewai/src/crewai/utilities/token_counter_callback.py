"""Token counting callback handler for LLM interactions.

This module provides a callback handler that tracks token usage
for LLM API calls. Works standalone and also integrates with litellm
when available (for the litellm fallback path).
"""

from typing import Any

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.utilities.logger_utils import suppress_warnings


# Check if litellm is available for callback integration
try:
    from litellm.integrations.custom_logger import CustomLogger as LiteLLMCustomLogger

    LITELLM_AVAILABLE = True
except ImportError:
    LiteLLMCustomLogger = None  # type: ignore[misc, assignment]
    LITELLM_AVAILABLE = False


# Create a base class that conditionally inherits from litellm's CustomLogger
# when available, or from object when not available
if LITELLM_AVAILABLE and LiteLLMCustomLogger is not None:
    _BaseClass: type = LiteLLMCustomLogger
else:
    _BaseClass = object


class TokenCalcHandler(_BaseClass):  # type: ignore[misc]
    """Handler for calculating and tracking token usage in LLM calls.

    This handler tracks prompt tokens, completion tokens, and cached tokens
    across requests. It works standalone and also integrates with litellm's
    logging system when litellm is installed (for the fallback path).

    Attributes:
        token_cost_process: The token process tracker to accumulate usage metrics.
    """

    def __init__(self, token_cost_process: TokenProcess | None, **kwargs: Any) -> None:
        """Initialize the token calculation handler.

        Args:
            token_cost_process: Optional token process tracker for accumulating metrics.
        """
        # Only call super().__init__ if we have a real parent class with __init__
        if LITELLM_AVAILABLE and LiteLLMCustomLogger is not None:
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

        This method has the same interface as litellm's CustomLogger.log_success_event()
        so it can be used as a litellm callback when litellm is installed, or called
        directly when litellm is not installed.

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

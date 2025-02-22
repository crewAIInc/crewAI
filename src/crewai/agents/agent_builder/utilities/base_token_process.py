from crewai.types.usage_metrics import UsageMetrics
import logging


logger = logging.getLogger(__name__)


class TokenProcess:
    def __init__(self) -> None:
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.cached_prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.successful_requests: int = 0

    def sum_prompt_tokens(self, tokens: int) -> None:
        self.prompt_tokens += tokens
        self.total_tokens += tokens

    def sum_completion_tokens(self, tokens: int) -> None:
        self.completion_tokens += tokens
        self.total_tokens += tokens

    def sum_cached_prompt_tokens(self, tokens: int | None) -> None:
        """
        Adds the given token count to cached prompt tokens.
        
        Args:
            tokens (int | None): Number of tokens to add. None values are ignored.
            
        Raises:
            ValueError: If tokens is negative.
        """
        if tokens is None:
            logger.debug("Received None value for token count")
            return
        if tokens < 0:
            raise ValueError("Token count cannot be negative")
        self.cached_prompt_tokens += tokens

    def sum_successful_requests(self, requests: int) -> None:
        self.successful_requests += requests

    def get_summary(self) -> UsageMetrics:
        return UsageMetrics(
            total_tokens=self.total_tokens,
            prompt_tokens=self.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            completion_tokens=self.completion_tokens,
            successful_requests=self.successful_requests,
        )

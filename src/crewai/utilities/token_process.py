"""Token processing utility for tracking and managing token usage."""

from crewai.types.usage_metrics import UsageMetrics

class TokenProcess:
    """Handles token processing and tracking for agents."""
    
    def __init__(self):
        """Initialize the token processor."""
        self._token_count = 0
        self._last_tokens = 0
    
    def update_token_count(self, count: int) -> None:
        """Update the token count.
        
        Args:
            count (int): Number of tokens to add to the count
        """
        self._token_count += count
        self._last_tokens = count
    
    def get_token_count(self) -> int:
        """Get the total token count.
        
        Returns:
            int: Total number of tokens processed
        """
        return self._token_count
    
    def get_last_tokens(self) -> int:
        """Get the number of tokens from the last update.
        
        Returns:
            int: Number of tokens from last update
        """
        return self._last_tokens
    
    def reset(self) -> None:
        """Reset the token counts to zero."""
        self._token_count = 0
        self._last_tokens = 0
        
    def get_summary(self) -> UsageMetrics:
        """Get a summary of token usage.
        
        Returns:
            UsageMetrics: Object containing token usage metrics
        """
        return UsageMetrics(
            total_tokens=self._token_count,
            prompt_tokens=0,  # These will be set by the LLM handler
            cached_prompt_tokens=0,
            completion_tokens=self._last_tokens,
            successful_requests=1 if self._token_count > 0 else 0
        )

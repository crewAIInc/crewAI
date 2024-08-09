from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """
    Model to track usage metrics for the crew's execution.

    Attributes:
        total_tokens: Total number of tokens used.
        prompt_tokens: Number of tokens used in prompts.
        completion_tokens: Number of tokens used in completions.
        successful_requests: Number of successful requests made.
    """

    total_tokens: int = Field(default=0, description="Total number of tokens used.")
    prompt_tokens: int = Field(
        default=0, description="Number of tokens used in prompts."
    )
    completion_tokens: int = Field(
        default=0, description="Number of tokens used in completions."
    )
    successful_requests: int = Field(
        default=0, description="Number of successful requests made."
    )

    def add_usage_metrics(self, usage_metrics: "UsageMetrics"):
        """
        Add the usage metrics from another UsageMetrics object.

        Args:
            usage_metrics (UsageMetrics): The usage metrics to add.
        """
        self.total_tokens += usage_metrics.total_tokens
        self.prompt_tokens += usage_metrics.prompt_tokens
        self.completion_tokens += usage_metrics.completion_tokens
        self.successful_requests += usage_metrics.successful_requests

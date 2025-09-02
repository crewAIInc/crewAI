class LLMQuotaLimitExceededException(Exception):
    QUOTA_LIMIT_ERRORS = [
        "quota exceeded",
        "rate limit exceeded",
        "resource exhausted",
        "too many requests",
        "quota limit reached",
        "api quota exceeded",
        "usage limit exceeded",
        "billing quota exceeded",
        "request limit exceeded",
        "daily quota exceeded",
        "monthly quota exceeded",
    ]

    def __init__(self, error_message: str):
        self.original_error_message = error_message
        super().__init__(self._get_error_message(error_message))

    def _is_quota_limit_error(self, error_message: str) -> bool:
        return any(
            phrase.lower() in error_message.lower()
            for phrase in self.QUOTA_LIMIT_ERRORS
        )

    def _get_error_message(self, error_message: str):
        return (
            f"LLM quota limit exceeded. Original error: {error_message}\n"
            "Your API quota or rate limit has been reached. Please check your API usage, "
            "upgrade your plan, or wait for the quota to reset before retrying."
        )

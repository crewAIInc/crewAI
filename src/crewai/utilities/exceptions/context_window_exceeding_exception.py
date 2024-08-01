class LLMContextLengthExceededException(Exception):
    CONTEXT_LIMIT_ERRORS = [
        "maximum context length",
        "context length exceeded",
        "context window full",
        "too many tokens",
        "input is too long",
        "exceeds token limit",
    ]

    def __init__(self, error_message: str):
        self.original_error_message = error_message
        if self._is_context_limit_error(error_message):
            super().__init__(self._get_error_message())
        else:
            raise ValueError(
                "The provided error message is not related to context length limits."
            )

    def _is_context_limit_error(self, error_message: str) -> bool:
        return any(
            phrase.lower() in error_message.lower()
            for phrase in self.CONTEXT_LIMIT_ERRORS
        )

    def _get_error_message(self):
        return (
            f"LLM context length exceeded. Original error: {self.original_error_message}\n"
            "Consider using a smaller input or implementing a text splitting strategy."
        )

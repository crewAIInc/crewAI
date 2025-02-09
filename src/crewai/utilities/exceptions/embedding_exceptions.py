from typing import List, Optional


class EmbeddingConfigurationError(Exception):
    def __init__(self, message: str, provider: Optional[str] = None):
        self.message = message
        self.provider = provider
        super().__init__(self.message)


class EmbeddingProviderError(EmbeddingConfigurationError):
    def __init__(self, provider: str, supported_providers: List[str]):
        message = f"Unsupported embedding provider: {provider}, supported providers: {supported_providers}"
        super().__init__(message, provider)


class EmbeddingInitializationError(EmbeddingConfigurationError):
    def __init__(self, provider: str, error: str):
        message = f"Failed to initialize embedding function for provider {provider}: {error}"
        super().__init__(message, provider)

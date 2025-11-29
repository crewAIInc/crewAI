"""Type definitions for HuggingFace embedding providers."""

from typing import Literal

from typing_extensions import Required, TypedDict


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace provider.

    Supports HuggingFace Inference API for text embeddings.

    Attributes:
        api_key: HuggingFace API key for authentication.
        model: Model name to use for embeddings (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        model_name: Alias for model.
        api_key_env_var: Environment variable name containing the API key.
        api_url: Optional API URL (accepted but not used, for compatibility).
        url: Alias for api_url (accepted but not used, for compatibility).
    """

    api_key: str
    model: str
    model_name: str
    api_key_env_var: str
    api_url: str
    url: str


class HuggingFaceProviderSpec(TypedDict, total=False):
    """HuggingFace provider specification."""

    provider: Required[Literal["huggingface"]]
    config: HuggingFaceProviderConfig

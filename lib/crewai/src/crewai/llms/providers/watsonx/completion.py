"""IBM watsonx.ai provider implementation.

This module provides native support for IBM Granite models via the
watsonx.ai Model Gateway, which exposes an OpenAI-compatible API.

Authentication uses IBM Cloud IAM token exchange: an API key is exchanged
for a short-lived Bearer token via the IAM identity service.

Usage:
    llm = LLM(model="watsonx/ibm/granite-4-h-small")
    llm = LLM(model="ibm/granite-4-h-small", provider="watsonx")

Environment variables:
    WATSONX_API_KEY: IBM Cloud API key (required)
    WATSONX_PROJECT_ID: watsonx.ai project ID (required)
    WATSONX_REGION: IBM Cloud region (default: us-south)
    WATSONX_URL: Full base URL override (optional)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx
from openai import OpenAI

from crewai.llms.providers.openai.completion import OpenAICompletion

logger = logging.getLogger(__name__)

# IBM Cloud IAM endpoint for token exchange
_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Default region for watsonx.ai
_DEFAULT_REGION = "us-south"

# Refresh token 60 seconds before expiry to avoid race conditions
_TOKEN_REFRESH_BUFFER_SECONDS = 60

# Supported watsonx.ai regions
_SUPPORTED_REGIONS = frozenset({
    "us-south",
    "eu-de",
    "eu-gb",
    "jp-tok",
    "au-syd",
})


class _IAMTokenManager:
    """Thread-safe IBM IAM token manager with automatic refresh.

    Exchanges an IBM Cloud API key for a short-lived Bearer token and
    caches it, refreshing automatically when the token approaches expiry.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._token: str | None = None
        self._expiry: float = 0.0
        self._lock = threading.Lock()

    def get_token(self) -> str:
        """Get a valid IAM Bearer token, refreshing if needed.

        Returns:
            A valid Bearer token string.

        Raises:
            RuntimeError: If the token exchange fails.
        """
        if self._token and time.time() < self._expiry - _TOKEN_REFRESH_BUFFER_SECONDS:
            return self._token

        with self._lock:
            # Double-check after acquiring lock
            if (
                self._token
                and time.time() < self._expiry - _TOKEN_REFRESH_BUFFER_SECONDS
            ):
                return self._token

            self._refresh_token()
            assert self._token is not None
            return self._token

    def _refresh_token(self) -> None:
        """Exchange API key for a new IAM token."""
        try:
            response = httpx.post(
                _IAM_TOKEN_URL,
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self._api_key,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"IBM IAM token exchange failed (HTTP {e.response.status_code}): "
                f"{e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"IBM IAM token exchange request failed: {e}"
            ) from e

        data = response.json()
        self._token = data["access_token"]
        self._expiry = float(data["expiration"])
        logger.debug(
            "IBM IAM token refreshed, expires at %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self._expiry)),
        )


class WatsonxCompletion(OpenAICompletion):
    """IBM watsonx.ai completion implementation.

    This class provides support for IBM Granite models and other foundation
    models hosted on watsonx.ai via the OpenAI-compatible Model Gateway.

    Authentication is handled transparently via IBM Cloud IAM token exchange.
    The API key is exchanged for a Bearer token which is automatically
    refreshed when it approaches expiry.

    Supported models include the IBM Granite family:
        - ibm/granite-4-h-small (32B hybrid)
        - ibm/granite-4-h-tiny (7B hybrid)
        - ibm/granite-4-h-micro (3B hybrid)
        - ibm/granite-3-8b-instruct
        - ibm/granite-3-3-8b-instruct
        - ibm/granite-8b-code-instruct
        - ibm/granite-guardian-3-8b
        - And other models available on watsonx.ai

    Example:
        # Using provider prefix
        llm = LLM(model="watsonx/ibm/granite-4-h-small")

        # Using explicit provider
        llm = LLM(model="ibm/granite-4-h-small", provider="watsonx")

        # With custom configuration
        llm = LLM(
            model="ibm/granite-4-h-small",
            provider="watsonx",
            api_key="my-ibm-cloud-api-key",
            temperature=0.7,
        )
    """

    def __init__(
        self,
        model: str,
        provider: str = "watsonx",
        api_key: str | None = None,
        base_url: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize watsonx.ai completion client.

        Args:
            model: The model identifier (e.g., "ibm/granite-4-h-small").
            provider: The provider name (default: "watsonx").
            api_key: IBM Cloud API key. If not provided, reads from
                WATSONX_API_KEY environment variable.
            base_url: Full base URL override for the watsonx.ai endpoint.
                If not provided, constructed from region.
            project_id: watsonx.ai project ID. If not provided, reads from
                WATSONX_PROJECT_ID environment variable.
            region: IBM Cloud region (default: "us-south"). If not provided,
                reads from WATSONX_REGION environment variable.
            **kwargs: Additional arguments passed to OpenAICompletion.

        Raises:
            ValueError: If required credentials are missing.
        """
        resolved_api_key = self._resolve_api_key(api_key)
        resolved_project_id = self._resolve_project_id(project_id)
        resolved_region = self._resolve_region(region)
        resolved_base_url = self._resolve_base_url(base_url, resolved_region)

        # Initialize IAM token manager for transparent auth
        self._iam_manager = _IAMTokenManager(resolved_api_key)
        self._project_id = resolved_project_id

        # Get initial token for client construction
        initial_token = self._iam_manager.get_token()

        # Pass the bearer token as api_key to OpenAI client
        # The OpenAI SDK uses this as Authorization: Bearer <token>
        super().__init__(
            model=model,
            provider=provider,
            api_key=initial_token,
            base_url=resolved_base_url,
            **kwargs,
        )

    @staticmethod
    def _resolve_api_key(api_key: str | None) -> str:
        """Resolve IBM Cloud API key from parameter or environment.

        Args:
            api_key: Explicitly provided API key.

        Returns:
            The resolved API key.

        Raises:
            ValueError: If no API key is found.
        """
        resolved = api_key or os.getenv("WATSONX_API_KEY")
        if not resolved:
            raise ValueError(
                "IBM Cloud API key is required for watsonx.ai provider. "
                "Set the WATSONX_API_KEY environment variable or pass "
                "api_key parameter."
            )
        return resolved

    @staticmethod
    def _resolve_project_id(project_id: str | None) -> str:
        """Resolve watsonx.ai project ID from parameter or environment.

        Args:
            project_id: Explicitly provided project ID.

        Returns:
            The resolved project ID.

        Raises:
            ValueError: If no project ID is found.
        """
        resolved = project_id or os.getenv("WATSONX_PROJECT_ID")
        if not resolved:
            raise ValueError(
                "watsonx.ai project ID is required. "
                "Set the WATSONX_PROJECT_ID environment variable or pass "
                "project_id parameter."
            )
        return resolved

    @staticmethod
    def _resolve_region(region: str | None) -> str:
        """Resolve IBM Cloud region from parameter or environment.

        Args:
            region: Explicitly provided region.

        Returns:
            The resolved region string.
        """
        resolved = region or os.getenv("WATSONX_REGION", _DEFAULT_REGION)
        if resolved not in _SUPPORTED_REGIONS:
            logger.warning(
                "Region '%s' is not in the known supported regions: %s. "
                "Proceeding anyway in case IBM has added new regions.",
                resolved,
                ", ".join(sorted(_SUPPORTED_REGIONS)),
            )
        return resolved

    @staticmethod
    def _resolve_base_url(base_url: str | None, region: str) -> str:
        """Resolve the watsonx.ai base URL.

        Priority:
            1. Explicit base_url parameter
            2. WATSONX_URL environment variable
            3. Constructed from region

        Args:
            base_url: Explicitly provided base URL.
            region: IBM Cloud region for URL construction.

        Returns:
            The resolved base URL.
        """
        if base_url:
            return base_url.rstrip("/")

        env_url = os.getenv("WATSONX_URL")
        if env_url:
            return env_url.rstrip("/")

        return f"https://{region}.ml.cloud.ibm.com/ml/v1"

    def _build_client(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> OpenAI:
        """Build the OpenAI client with watsonx-specific configuration.

        Overrides the parent method to inject the project_id header
        and ensure the IAM token is current.

        Args:
            api_key: Bearer token (from IAM exchange).
            base_url: watsonx.ai endpoint URL.
            default_headers: Additional headers.

        Returns:
            Configured OpenAI client instance.
        """
        # Refresh token if needed
        current_token = self._iam_manager.get_token()

        # Merge watsonx-specific headers
        watsonx_headers = {
            "X-Watsonx-Project-Id": self._project_id,
        }
        if default_headers:
            watsonx_headers.update(default_headers)

        return super()._build_client(
            api_key=current_token,
            base_url=base_url,
            default_headers=watsonx_headers,
        )

    def _ensure_fresh_token(self) -> None:
        """Refresh the IAM token on the client if needed.

        Updates the client's API key (Bearer token) if the cached
        token has been refreshed.
        """
        current_token = self._iam_manager.get_token()
        if hasattr(self, "client") and self.client is not None:
            self.client.api_key = current_token

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        """Call the LLM, refreshing the IAM token if needed."""
        self._ensure_fresh_token()
        return super().call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                    from_task=None, from_agent=None, response_model=None):
        """Async call the LLM, refreshing the IAM token if needed."""
        self._ensure_fresh_token()
        return await super().acall(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    def get_context_window_size(self) -> int:
        """Get context window size for Granite models.

        Returns:
            The context window size in tokens.
        """
        model_lower = self.model.lower()

        # Granite 4.x models have 128K context
        if "granite-4" in model_lower:
            return 131072

        # Granite 3.x instruct models have 128K context
        if "granite-3" in model_lower and "instruct" in model_lower:
            return 131072

        # Granite 3.x base models have 4K context
        if "granite-3" in model_lower:
            return 4096

        # Granite code models
        if "granite" in model_lower and "code" in model_lower:
            return 8192

        # Default for unknown models
        return 8192

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling / tool use.

        Granite 3.x instruct and 4.x models support tool use.

        Returns:
            True if the model supports function calling.
        """
        model_lower = self.model.lower()

        # Granite 4.x models support tool use
        if "granite-4" in model_lower:
            return True

        # Granite 3.x instruct models support tool use
        if "granite-3" in model_lower and "instruct" in model_lower:
            return True

        # Granite guardian models don't do tool use
        if "guardian" in model_lower:
            return False

        # Default: assume no tool use for unknown models
        return False

    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs.

        Currently, Granite models are text-only.

        Returns:
            False (Granite models are text-only).
        """
        return False

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize this LLM to a dict for reconstruction.

        Returns:
            Configuration dict with watsonx-specific fields.
        """
        config = super().to_config_dict()
        config["model"] = f"watsonx/{self.model}" if "/" not in self.model else f"watsonx/{self.model}"
        return config

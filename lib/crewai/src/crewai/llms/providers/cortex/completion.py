"""Snowflake Cortex native LLM provider for CrewAI.

Provides direct integration with the Snowflake Cortex REST API for
text completion and tool calling without requiring LiteLLM.

Authentication (checked in order):
    1. PAT: Set SNOWFLAKE_PAT env var or pass pat= kwarg
    2. SPCS: Auto-detected when running in Snowpark Container Services
    3. JWT key-pair: Set SNOWFLAKE_PRIVATE_KEY_PATH + SNOWFLAKE_USER env vars
       (requires ``crewai[cortex]`` extra for the ``cryptography`` package)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context


if TYPE_CHECKING:
    from crewai.utilities.types import LLMMessage

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for the Cortex provider but is not installed."
    ) from None

# Models that support tool/function calling on Cortex
CORTEX_TOOL_CALLING_MODELS: Final[tuple[str, ...]] = (
    "claude-3-5-sonnet",
    "claude-3-7-sonnet",
    "claude-sonnet-4",
    "claude-opus-4",
)

# Context window sizes (tokens) for Cortex-hosted models
CORTEX_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "claude-3-5-sonnet": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "deepseek-r1": 128_000,
    "llama3.1-8b": 128_000,
    "llama3.1-70b": 128_000,
    "llama3.1-405b": 128_000,
    "llama3.2-1b": 128_000,
    "llama3.2-3b": 128_000,
    "llama3.3-70b": 128_000,
    "llama4-maverick": 1_048_576,
    "llama4-scout": 512_000,
    "mistral-7b": 32_000,
    "mistral-large": 128_000,
    "mistral-large2": 128_000,
    "mixtral-8x7b": 32_000,
    "reka-core": 128_000,
    "reka-flash": 128_000,
    "jamba-instruct": 256_000,
    "jamba-1.5-mini": 256_000,
    "jamba-1.5-large": 256_000,
    "gemma-7b": 8_192,
    "snowflake-arctic": 4_096,
}


def _get_auth_token(
    account: str,
    user: str | None = None,
    pat: str | None = None,
    private_key_path: str | None = None,
    private_key: str | None = None,
) -> tuple[str, str]:
    """Resolve an authentication token for the Cortex REST API.

    Tries auth methods in order: PAT -> SPCS auto-detect -> JWT key-pair.

    Returns:
        Tuple of (token, token_type) where token_type is the value for
        the ``X-Snowflake-Authorization-Token-Type`` header.
    """
    # 1. Personal Access Token
    if pat:
        return pat, "PROGRAMMATIC_ACCESS_TOKEN"

    # 2. SPCS auto-detect
    spcs_token_path = "/snowflake/session/token"  # noqa: S105
    if os.path.isfile(spcs_token_path):
        with open(spcs_token_path) as f:
            return f.read().strip(), "OAUTH"

    # 3. JWT key-pair
    if private_key_path or private_key:
        if not user:
            raise ValueError(
                "SNOWFLAKE_USER is required for JWT key-pair authentication"
            )
        return _generate_jwt(
            account, user, private_key_path, private_key
        ), "KEYPAIR_JWT"

    raise ValueError(
        "No Snowflake authentication configured. Set one of:\n"
        "  - SNOWFLAKE_PAT (Personal Access Token)\n"
        "  - SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PRIVATE_KEY (JWT key-pair)\n"
        "  - Run inside Snowpark Container Services (auto-detected)"
    )


def _generate_jwt(
    account: str,
    user: str,
    private_key_path: str | None = None,
    private_key: str | None = None,
) -> str:
    """Generate a Snowflake JWT for key-pair authentication."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        import jwt
    except ImportError:
        raise ImportError(
            "JWT key-pair auth requires the cryptography package.\n"
            'Install with: uv add "crewai[cortex]"'
        ) from None

    import base64
    from datetime import datetime, timedelta, timezone
    import hashlib

    if private_key_path:
        with open(private_key_path, "rb") as f:
            key_bytes = f.read()
    elif private_key:
        key_bytes = private_key.encode("utf-8")
    else:
        raise ValueError("Either private_key_path or private_key must be provided")

    key = load_pem_private_key(key_bytes, password=None)

    # Compute public key fingerprint
    public_key_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    sha256_digest = hashlib.sha256(public_key_der).digest()
    fingerprint = "SHA256:" + base64.b64encode(sha256_digest).decode("utf-8")

    # Build qualified username (strip region/cloud suffix)
    account_upper = account.upper().split(".")[0]
    user_upper = user.upper()
    qualified_username = f"{account_upper}.{user_upper}"

    now = datetime.now(timezone.utc)
    payload = {
        "iss": f"{qualified_username}.{fingerprint}",
        "sub": qualified_username,
        "iat": now,
        "exp": now + timedelta(hours=1),
    }

    return jwt.encode(payload, key, algorithm="RS256")


class CortexCompletion(BaseLLM):
    """Snowflake Cortex native completion implementation.

    Provides direct integration with the Snowflake Cortex Complete REST API,
    supporting text completion and tool calling for compatible models.

    Usage::

        llm = LLM(model="cortex/claude-3-5-sonnet")
        # or
        llm = LLM(model="llama3.1-70b", provider="cortex")
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        account: str | None = None,
        user: str | None = None,
        pat: str | None = None,
        private_key_path: str | None = None,
        private_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        top_p: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        """Initialize Snowflake Cortex completion client.

        Args:
            model: Cortex model name (e.g., ``claude-3-5-sonnet``, ``llama3.1-70b``)
            account: Snowflake account identifier (defaults to SNOWFLAKE_ACCOUNT env var)
            user: Snowflake username (defaults to SNOWFLAKE_USER env var)
            pat: Personal Access Token (defaults to SNOWFLAKE_PAT env var)
            private_key_path: Path to PEM private key (defaults to SNOWFLAKE_PRIVATE_KEY_PATH)
            private_key: PEM private key string (defaults to SNOWFLAKE_PRIVATE_KEY)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            api_key: Alias for ``pat`` (compatibility with CrewAI factory)
            base_url: Override the Cortex API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retries on transient errors
            **kwargs: Additional parameters passed to BaseLLM
        """
        super().__init__(model=model, temperature=temperature, **kwargs)

        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        if not self.account:
            raise ValueError(
                "Snowflake account is required. Set SNOWFLAKE_ACCOUNT env var "
                "or pass account= to the constructor."
            )

        self.user = user or os.getenv("SNOWFLAKE_USER")
        self.pat = pat or api_key or os.getenv("SNOWFLAKE_PAT")
        self.private_key_path = private_key_path or os.getenv(
            "SNOWFLAKE_PRIVATE_KEY_PATH"
        )
        self.private_key = private_key or os.getenv("SNOWFLAKE_PRIVATE_KEY")

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = f"https://{self.account}.snowflakecomputing.com"

        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries

        # Resolve auth
        self._token, self._token_type = _get_auth_token(
            account=self.account,
            user=self.user,
            pat=self.pat,
            private_key_path=self.private_key_path,
            private_key=self.private_key,
        )

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._build_headers(),
        )

    def close(self) -> None:
        """Close the underlying httpx client to release connections."""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self) -> None:
        self.close()

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
            "X-Snowflake-Authorization-Token-Type": self._token_type,
        }

    def _refresh_token(self) -> None:
        """Refresh the auth token (JWT tokens expire after 1 hour)."""
        self._token, self._token_type = _get_auth_token(
            account=self.account,
            user=self.user,
            pat=self.pat,
            private_key_path=self.private_key_path,
            private_key=self.private_key,
        )
        self.client.headers.update(self._build_headers())

    def _supports_tool_calling(self) -> bool:
        """Check if the current model supports tool calling."""
        model_lower = self.model.lower()
        return any(tc in model_lower for tc in CORTEX_TOOL_CALLING_MODELS)

    def supports_stop_words(self) -> bool:
        """Cortex supports stop words based on configuration."""
        return self._supports_stop_words_implementation()

    def get_context_window_size(self) -> int:
        """Get context window size for the current model."""
        return CORTEX_CONTEXT_WINDOWS.get(self.model, 128_000)

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call Snowflake Cortex Complete API.

        Args:
            messages: Input messages
            tools: Tool definitions for function calling
            callbacks: Callback functions (unused in native implementation)
            available_functions: Available functions for tool execution
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Optional Pydantic model for structured output

        Returns:
            Text response or tool call result
        """
        with llm_call_context():
            try:
                self._emit_call_started_event(
                    messages=messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                formatted = self._format_messages(messages)
                payload = self._build_request_payload(formatted, tools)
                response_data = self._make_request(payload)

                return self._process_response(
                    response_data,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            except Exception as e:
                error_msg = f"Cortex API call failed: {e!s}"
                logging.error(error_msg)
                self._emit_call_failed_event(
                    error=error_msg, from_task=from_task, from_agent=from_agent
                )
                raise

    def _build_request_payload(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the Cortex Complete API request payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if tools and self._supports_tool_calling():
            payload["tools"] = self._convert_tools(tools)

        return payload

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to Cortex tool_spec format.

        Cortex expects: ``{"tool_spec": {"type": "function", "function": {...}}}``
        """
        cortex_tools = []
        for tool in tools:
            # Already in Cortex format
            if "tool_spec" in tool:
                cortex_tools.append(tool)
                continue

            try:
                from crewai.llms.providers.utils.common import safe_tool_conversion

                name, description, parameters = safe_tool_conversion(tool, "Cortex")
            except (ImportError, KeyError, ValueError) as e:
                logging.error(f"Error converting tool for Cortex: {e}")
                raise

            cortex_tools.append(
                {
                    "tool_spec": {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters
                            or {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        },
                    }
                }
            )

        return cortex_tools

    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to Cortex API with retry logic."""
        url = "/api/v2/cortex/inference:complete"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(url, json=payload)

                # Refresh token and retry on 401 — only useful for JWT
                # (PAT/SPCS tokens don't refresh, so fail fast)
                if response.status_code == 401 and attempt < self.max_retries:
                    if self._token_type == "KEYPAIR_JWT":
                        self._refresh_token()
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(
                    f"Cortex API error (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                ) from e
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                raise RuntimeError(f"Cortex API timeout after {self.timeout}s") from e

        raise RuntimeError(
            f"Cortex API call failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _process_response(
        self,
        response_data: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Process the Cortex API response.

        Handles both text responses and tool call responses.
        Cortex returns ``choices[0]["messages"]`` which is either a plain string
        (text completion) or a dict with ``role``, ``content``, and optionally
        ``tool_calls``.
        """
        usage = response_data.get("usage", {})
        if usage:
            self._track_token_usage_internal(usage)

        choices = response_data.get("choices", [])
        if not choices:
            raise RuntimeError("Cortex API returned empty choices")

        message = choices[0].get("messages", "")

        # Simple text response
        if isinstance(message, str):
            content = self._apply_stop_words(message)
            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
            )
            if response_model:
                return self._validate_structured_output(content, response_model)
            return content

        # Dict response — may contain tool_calls
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])

            if tool_calls and available_functions:
                tool_call = tool_calls[0]
                function_info = tool_call.get("function", {})
                function_name = function_info.get("name", "")

                try:
                    function_args = json.loads(function_info.get("arguments", "{}"))
                except json.JSONDecodeError:
                    function_args = {}

                result = self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                if result is not None:
                    return result

            # Fall back to text content
            content = message.get("content", "")
            content = self._apply_stop_words(content)
            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
            )
            if response_model:
                return self._validate_structured_output(content, response_model)
            return content

        raise RuntimeError(f"Unexpected Cortex response format: {type(message)}")

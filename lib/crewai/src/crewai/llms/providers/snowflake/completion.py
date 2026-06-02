from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import model_validator

from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.types import LLMMessage


SNOWFLAKE_CORTEX_PATH = "/api/v2/cortex/v1"
SNOWFLAKE_TOKEN_ENV_VARS = (
    "SNOWFLAKE_PAT",
    "SNOWFLAKE_TOKEN",
    "SNOWFLAKE_JWT",
)


def _normalize_snowflake_base_url(value: str) -> str:
    """Return a Snowflake Cortex REST OpenAI-compatible base URL."""
    base_url = value.strip().rstrip("/")
    if not base_url:
        raise ValueError("Snowflake account URL cannot be empty")

    if "://" not in base_url:
        base_url = f"https://{base_url}"

    if base_url.endswith(SNOWFLAKE_CORTEX_PATH):
        return base_url

    if "/api/v2/cortex" in base_url:
        raise ValueError(
            "Snowflake base URL must be the account URL or Cortex API root "
            f"ending in {SNOWFLAKE_CORTEX_PATH}; do not include endpoint paths."
        )

    return f"{base_url}{SNOWFLAKE_CORTEX_PATH}"


def _base_url_from_account_identifier(account_identifier: str) -> str:
    account = account_identifier.strip()
    if not account:
        raise ValueError("Snowflake account identifier cannot be empty")
    return _normalize_snowflake_base_url(f"{account}.snowflakecomputing.com")


class SnowflakeCompletion(OpenAICompletion):
    """Snowflake Cortex REST API native completion implementation.

    Snowflake exposes an OpenAI-compatible Chat Completions endpoint at
    ``/api/v2/cortex/v1/chat/completions``. This provider reuses CrewAI's
    native OpenAI transport while applying Snowflake-specific authentication,
    endpoint normalization, and Claude-family message constraints.
    """

    provider: str = "snowflake"
    api: Literal["completions"] = "completions"
    account_url: str | None = None
    account_identifier: str | None = None
    database: str | None = None
    schema_name: str | None = None
    warehouse: str | None = None
    role: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_snowflake_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data["provider"] = "snowflake"
        api = data.get("api")
        if api and api != "completions":
            raise ValueError(
                "Snowflake Cortex native provider supports only the Chat Completions API"
            )
        data["api"] = "completions"

        data["api_key"] = cls._resolve_token(data.get("api_key"))
        resolved_base_url = cls._resolve_base_url(data)
        data["base_url"] = resolved_base_url
        data["account_url"] = resolved_base_url

        return data

    @staticmethod
    def _resolve_token(api_key: str | None) -> str:
        token = api_key
        if not token:
            for env_var in SNOWFLAKE_TOKEN_ENV_VARS:
                token = os.getenv(env_var)
                if token:
                    break

        if not token:
            raise ValueError(
                "Snowflake token is required. Set SNOWFLAKE_PAT, SNOWFLAKE_TOKEN, "
                "or SNOWFLAKE_JWT, or pass api_key."
            )

        if token.startswith("pat/"):
            token = token.removeprefix("pat/")

        return token

    @classmethod
    def _resolve_base_url(cls, data: dict[str, Any]) -> str:
        explicit_base_url = data.get("base_url") or data.get("api_base")
        if explicit_base_url:
            return _normalize_snowflake_base_url(explicit_base_url)

        account_url = data.get("account_url") or os.getenv("SNOWFLAKE_ACCOUNT_URL")
        if account_url:
            return _normalize_snowflake_base_url(account_url)

        account_identifier = (
            data.get("account_identifier")
            or data.get("account")
            or data.get("snowflake_account")
            or os.getenv("SNOWFLAKE_ACCOUNT")
            or os.getenv("SNOWFLAKE_ACCOUNT_ID")
            or os.getenv("SNOWFLAKE_ACCOUNT_IDENTIFIER")
        )
        if account_identifier:
            return _base_url_from_account_identifier(account_identifier)

        raise ValueError(
            "Snowflake account URL is required. Set SNOWFLAKE_ACCOUNT_URL or "
            "SNOWFLAKE_ACCOUNT, or pass account_url/base_url/account_identifier."
        )

    def _format_messages(self, messages: str | list[LLMMessage]) -> list[LLMMessage]:
        formatted_messages = super()._format_messages(messages)
        if self._is_claude_model():
            formatted_messages = self._remove_incomplete_claude_tool_uses(
                formatted_messages
            )
            return self._ensure_claude_conversation_ends_with_user(formatted_messages)
        return formatted_messages

    def _is_claude_model(self) -> bool:
        model = self.model.lower()
        return model.startswith(("claude-", "anthropic."))

    @staticmethod
    def _remove_incomplete_claude_tool_uses(
        messages: list[LLMMessage],
    ) -> list[LLMMessage]:
        """Drop dangling Claude tool-use turns before sending to Snowflake.

        Snowflake-hosted Claude models reject histories where an assistant tool
        use is not accompanied by matching tool results. CrewAI may retry or
        summarize after an interrupted tool cycle, leaving an assistant
        ``tool_calls`` message in history without every corresponding
        ``role='tool'`` result. OpenAI-family models tolerate that more often,
        but Claude through Snowflake returns:
        "Each 'toolUse' block must be accompanied with a matching 'toolResult' block."
        """
        sanitized: list[LLMMessage] = []
        index = 0

        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not tool_calls:
                sanitized.append(message)
                index += 1
                continue

            expected_ids = {
                tool_call.get("id")
                for tool_call in tool_calls
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            if not expected_ids:
                sanitized.append(message)
                index += 1
                continue

            tool_result_ids: set[str] = set()
            lookahead = index + 1
            while (
                lookahead < len(messages) and messages[lookahead].get("role") == "tool"
            ):
                tool_call_id = messages[lookahead].get("tool_call_id")
                if isinstance(tool_call_id, str):
                    tool_result_ids.add(tool_call_id)
                lookahead += 1

            if expected_ids.issubset(tool_result_ids):
                sanitized.append(message)
                sanitized.extend(messages[index + 1 : lookahead])

            index = lookahead

        return sanitized

    @staticmethod
    def _ensure_claude_conversation_ends_with_user(
        messages: list[LLMMessage],
    ) -> list[LLMMessage]:
        if not messages:
            return [{"role": "user", "content": "Hello"}]

        if messages[-1].get("role") == "assistant" and not messages[-1].get(
            "tool_calls"
        ):
            messages = messages[:-1]

        if not messages:
            return [{"role": "user", "content": "Hello"}]

        if messages[-1].get("role") == "user":
            return messages

        return [
            *messages,
            {
                "role": "user",
                "content": "Please continue and provide your final answer.",
            },
        ]

    def _prepare_completion_params(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        params = super()._prepare_completion_params(messages=messages, tools=tools)
        if self._is_claude_model() and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        return params

    def supports_function_calling(self) -> bool:
        model = self.model.lower()
        return model.startswith(("openai-", "claude-", "anthropic."))

    def supports_multimodal(self) -> bool:
        model = self.model.lower()
        return model.startswith(("openai-", "claude-", "anthropic."))

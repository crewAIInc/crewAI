"""OpenAI-compatible providers implementation.

This module provides a thin subclass of OpenAICompletion that supports
various OpenAI-compatible APIs like OpenRouter, DeepSeek, Ollama, vLLM,
Cerebras, and Dashscope (Alibaba/Qwen).

Usage:
    llm = LLM(model="deepseek/deepseek-chat")  # Uses DeepSeek API
    llm = LLM(model="openrouter/anthropic/claude-3-opus")  # Uses OpenRouter
    llm = LLM(model="ollama/llama3")  # Uses local Ollama
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, model_validator

from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an OpenAI-compatible provider.

    Attributes:
        base_url: Default base URL for the provider's API endpoint.
        api_key_env: Environment variable name for the API key.
        base_url_env: Environment variable name for a custom base URL override.
        default_headers: HTTP headers to include in all requests.
        api_key_required: Whether an API key is required for this provider.
        default_api_key: Default API key to use if none is provided and not required.
        supports_json_schema: Whether the provider supports json_schema response_format type.
    """

    base_url: str
    api_key_env: str
    base_url_env: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    api_key_required: bool = True
    default_api_key: str | None = None
    supports_json_schema: bool = True


OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        base_url_env="OPENROUTER_BASE_URL",
        default_headers={"HTTP-Referer": "https://crewai.com"},
        api_key_required=True,
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        api_key_required=True,
        supports_json_schema=False,
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        base_url_env="OLLAMA_HOST",
        api_key_required=False,
        default_api_key="ollama",
    ),
    "ollama_chat": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        base_url_env="OLLAMA_HOST",
        api_key_required=False,
        default_api_key="ollama",
    ),
    "hosted_vllm": ProviderConfig(
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        base_url_env="VLLM_BASE_URL",
        api_key_required=False,
        default_api_key="dummy",
    ),
    "cerebras": ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        base_url_env="CEREBRAS_BASE_URL",
        api_key_required=True,
    ),
    "dashscope": ProviderConfig(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        api_key_required=True,
    ),
}


def _normalize_ollama_base_url(base_url: str) -> str:
    """Normalize Ollama base URL to ensure it ends with /v1.

    Ollama uses OLLAMA_HOST which may not include the /v1 suffix,
    but the OpenAI-compatible endpoint requires it.

    Args:
        base_url: The base URL, potentially without /v1 suffix.

    Returns:
        The base URL with /v1 suffix if needed.
    """
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        return f"{base_url}/v1"
    return base_url


class OpenAICompatibleCompletion(OpenAICompletion):
    """OpenAI-compatible completion implementation.

    This class provides support for various OpenAI-compatible APIs by
    automatically configuring the base URL, API key, and headers based
    on the provider name.

    Supported providers:
        - openrouter: OpenRouter (https://openrouter.ai)
        - deepseek: DeepSeek (https://deepseek.com)
        - ollama: Ollama local server (https://ollama.ai)
        - ollama_chat: Alias for ollama
        - hosted_vllm: vLLM server (https://github.com/vllm-project/vllm)
        - cerebras: Cerebras (https://cerebras.ai)
        - dashscope: Alibaba Dashscope/Qwen (https://dashscope.aliyun.com)

    Example:
        # Using provider prefix
        llm = LLM(model="deepseek/deepseek-chat")

        # Using explicit provider parameter
        llm = LLM(model="llama3", provider="ollama")

        # With custom configuration
        llm = LLM(
            model="deepseek-chat",
            provider="deepseek",
            api_key="my-key",
            temperature=0.7
        )
    """

    @model_validator(mode="before")
    @classmethod
    def _resolve_provider_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        provider = data.get("provider", "")
        config = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if config is None:
            supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS.keys()))
            raise ValueError(
                f"Unknown OpenAI-compatible provider: {provider}. "
                f"Supported providers: {supported}"
            )

        data["api_key"] = cls._resolve_api_key(data.get("api_key"), config, provider)
        data["base_url"] = cls._resolve_base_url(data.get("base_url"), config, provider)
        data["default_headers"] = cls._resolve_headers(
            data.get("default_headers"), config
        )
        return data

    @staticmethod
    def _resolve_api_key(
        api_key: str | None,
        config: ProviderConfig,
        provider: str,
    ) -> str | None:
        """Resolve the API key from explicit value, env var, or default.

        Args:
            api_key: Explicitly provided API key.
            config: Provider configuration.
            provider: Provider name for error messages.

        Returns:
            The resolved API key.

        Raises:
            ValueError: If API key is required but not found.
        """
        if api_key:
            return api_key

        env_key = os.getenv(config.api_key_env)
        if env_key:
            return env_key

        if config.api_key_required:
            raise ValueError(
                f"API key required for {provider}. "
                f"Set {config.api_key_env} environment variable or pass api_key parameter."
            )

        return config.default_api_key

    @staticmethod
    def _resolve_base_url(
        base_url: str | None,
        config: ProviderConfig,
        provider: str,
    ) -> str:
        """Resolve the base URL from explicit value, env var, or default.

        Args:
            base_url: Explicitly provided base URL.
            config: Provider configuration.
            provider: Provider name (used for special handling like Ollama).

        Returns:
            The resolved base URL.
        """
        if base_url:
            resolved = base_url
        elif config.base_url_env:
            env_value = os.getenv(config.base_url_env)
            resolved = env_value if env_value else config.base_url
        else:
            resolved = config.base_url

        if provider in ("ollama", "ollama_chat"):
            resolved = _normalize_ollama_base_url(resolved)

        return resolved

    @staticmethod
    def _resolve_headers(
        headers: dict[str, str] | None,
        config: ProviderConfig,
    ) -> dict[str, str] | None:
        """Merge user headers with provider default headers.

        Args:
            headers: User-provided headers.
            config: Provider configuration.

        Returns:
            Merged headers dict, or None if empty.
        """
        if not config.default_headers and not headers:
            return None

        merged = dict(config.default_headers)
        if headers:
            merged.update(headers)

        return merged if merged else None

    @property
    def _provider_supports_json_schema(self) -> bool:
        """Check if the current provider supports json_schema response_format."""
        config = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        if config is None:
            return True
        return config.supports_json_schema

    def _prepare_completion_params(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
    ) -> dict[str, Any]:
        """Prepare params, stripping json_schema response_format if unsupported."""
        params = super()._prepare_completion_params(messages, tools)

        if not self._provider_supports_json_schema:
            rf = params.get("response_format")
            if isinstance(rf, dict) and rf.get("type") == "json_schema":
                schema_info = rf.get("json_schema", {})
                schema = schema_info.get("schema", schema_info)
                self._inject_schema_instructions(params, schema)
                del params["response_format"]

        return params

    def _inject_schema_instructions(
        self,
        params: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        """Inject JSON schema instructions into the system message."""
        schema_str = json.dumps(schema, indent=2)
        instruction = (
            "\nYou must respond with a valid JSON object that conforms to this JSON schema:\n"
            f"```json\n{schema_str}\n```\n"
            "Respond ONLY with valid JSON, no additional text or markdown."
        )
        msgs = params.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "system":
                msg["content"] = (msg.get("content") or "") + instruction
                return
        params["messages"] = [
            {"role": "system", "content": instruction.lstrip()},
            *msgs,
        ]

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """Extract JSON from text that may be wrapped in markdown code blocks."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            json_lines: list[str] = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    json_lines.append(line)
            return "\n".join(json_lines).strip()
        return stripped

    def _handle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle completion, falling back for providers without json_schema."""
        if response_model and not self._provider_supports_json_schema:
            return self._handle_completion_fallback(
                params, available_functions, from_task, from_agent, response_model
            )
        return super()._handle_completion(
            params, available_functions, from_task, from_agent, response_model
        )

    def _handle_completion_fallback(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle structured output via prompt injection instead of json_schema."""
        from crewai.events.types.llm_events import LLMCallType

        schema_dict = response_model.model_json_schema() if response_model else {}
        modified_params = dict(params)
        modified_params.pop("response_format", None)

        self._inject_schema_instructions(modified_params, schema_dict)

        response = self._get_sync_client().chat.completions.create(**modified_params)

        usage = self._extract_openai_token_usage(response)
        self._track_token_usage_internal(usage)

        message = response.choices[0].message

        if message.tool_calls and not available_functions:
            self._emit_call_completed_event(
                response=list(message.tool_calls),
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=modified_params["messages"],
                usage=usage,
            )
            return list(message.tool_calls)

        content = message.content or ""
        if response_model:
            try:
                json_content = self._extract_json_from_text(content)
                parsed = response_model.model_validate_json(json_content)
                self._emit_call_completed_event(
                    response=parsed.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=modified_params["messages"],
                    usage=usage,
                )
                return parsed
            except Exception as e:
                logging.warning(
                    f"Structured output parsing failed, returning raw content: {e}"
                )

        content = self._apply_stop_words(content)
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=modified_params["messages"],
            usage=usage,
        )
        return content

    async def _ahandle_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async completion, falling back for providers without json_schema."""
        if response_model and not self._provider_supports_json_schema:
            return await self._ahandle_completion_fallback(
                params, available_functions, from_task, from_agent, response_model
            )
        return await super()._ahandle_completion(
            params, available_functions, from_task, from_agent, response_model
        )

    async def _ahandle_completion_fallback(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle async structured output via prompt injection instead of json_schema."""
        from crewai.events.types.llm_events import LLMCallType

        schema_dict = response_model.model_json_schema() if response_model else {}
        modified_params = dict(params)
        modified_params.pop("response_format", None)

        self._inject_schema_instructions(modified_params, schema_dict)

        response = await self._get_async_client().chat.completions.create(
            **modified_params
        )

        usage = self._extract_openai_token_usage(response)
        self._track_token_usage_internal(usage)

        message = response.choices[0].message

        if message.tool_calls and not available_functions:
            self._emit_call_completed_event(
                response=list(message.tool_calls),
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=modified_params["messages"],
                usage=usage,
            )
            return list(message.tool_calls)

        content = message.content or ""
        if response_model:
            try:
                json_content = self._extract_json_from_text(content)
                parsed = response_model.model_validate_json(json_content)
                self._emit_call_completed_event(
                    response=parsed.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=modified_params["messages"],
                    usage=usage,
                )
                return parsed
            except Exception as e:
                logging.warning(
                    f"Structured output parsing failed, returning raw content: {e}"
                )

        content = self._apply_stop_words(content)
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=modified_params["messages"],
            usage=usage,
        )
        return content

    def _handle_streaming_completion(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | list[dict[str, Any]] | Any:
        """Handle streaming completion, falling back for providers without json_schema."""
        if response_model and not self._provider_supports_json_schema:
            return self._handle_streaming_completion_fallback(
                params, available_functions, from_task, from_agent, response_model
            )
        return super()._handle_streaming_completion(
            params, available_functions, from_task, from_agent, response_model
        )

    def _handle_streaming_completion_fallback(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Handle streaming structured output via prompt injection."""
        from crewai.events.types.llm_events import LLMCallType

        schema_dict = response_model.model_json_schema() if response_model else {}
        modified_params = dict(params)
        modified_params.pop("response_format", None)

        self._inject_schema_instructions(modified_params, schema_dict)

        full_response = ""
        usage_data: dict[str, Any] | None = None

        completion_stream = self._get_sync_client().chat.completions.create(
            **modified_params
        )

        for chunk in completion_stream:
            response_id_stream = chunk.id if hasattr(chunk, "id") else None

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = self._extract_openai_token_usage(chunk)
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                full_response += delta.content
                self._emit_stream_chunk_event(
                    chunk=delta.content,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_id=response_id_stream,
                )

        if usage_data:
            self._track_token_usage_internal(usage_data)

        if response_model:
            try:
                json_content = self._extract_json_from_text(full_response)
                parsed = response_model.model_validate_json(json_content)
                self._emit_call_completed_event(
                    response=parsed.model_dump_json(),
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=modified_params["messages"],
                    usage=usage_data,
                )
                return parsed
            except Exception as e:
                logging.warning(f"Structured output parsing failed in stream: {e}")

        self._emit_call_completed_event(
            response=full_response,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=modified_params["messages"],
            usage=usage_data,
        )
        return full_response

    def supports_function_calling(self) -> bool:
        """Check if the provider supports function calling.

        Delegates to the parent OpenAI implementation which handles
        edge cases like o1 models (which may be routed through
        OpenRouter or other compatible providers).

        Returns:
            Whether the model supports function calling.
        """
        return super().supports_function_calling()

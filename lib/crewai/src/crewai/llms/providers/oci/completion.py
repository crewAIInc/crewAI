from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.utilities.oci import create_oci_client_kwargs, get_oci_module
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"
DEFAULT_OCI_REGION = "us-chicago-1"


def _get_oci_module() -> Any:
    """Backward-compatible module-local alias used by tests and patches."""
    return get_oci_module()


class OCICompletion(BaseLLM):
    """OCI Generative AI native provider for CrewAI.

    Supports basic text completions for generic (Meta, Google, OpenAI, xAI)
    and Cohere model families hosted on the OCI Generative AI service.
    """

    def __init__(
        self,
        model: str,
        *,
        compartment_id: str | None = None,
        service_endpoint: str | None = None,
        auth_type: Literal[
            "API_KEY",
            "SECURITY_TOKEN",
            "INSTANCE_PRINCIPAL",
            "RESOURCE_PRINCIPAL",
        ]
        | str = "API_KEY",
        auth_profile: str | None = None,
        auth_file_location: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        oci_provider: str | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("provider", None)
        super().__init__(
            model=model,
            temperature=temperature,
            provider="oci",
            **kwargs,
        )

        self.compartment_id = compartment_id or os.getenv("OCI_COMPARTMENT_ID")
        if not self.compartment_id:
            raise ValueError(
                "OCI compartment_id is required. Set compartment_id or OCI_COMPARTMENT_ID."
            )

        self.service_endpoint = service_endpoint or os.getenv("OCI_SERVICE_ENDPOINT")
        if self.service_endpoint is None:
            region = os.getenv("OCI_REGION", DEFAULT_OCI_REGION)
            self.service_endpoint = (
                f"https://inference.generativeai.{region}.oci.oraclecloud.com"
            )

        self.auth_type = str(auth_type).upper()
        self.auth_profile = cast(
            str, auth_profile or os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
        )
        self.auth_file_location = cast(
            str,
            auth_file_location
            or os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
        )
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.oci_provider = oci_provider or self._infer_provider(model)
        self._oci = _get_oci_module()

        if client is not None:
            self.client = client
        else:
            client_kwargs = create_oci_client_kwargs(
                auth_type=self.auth_type,
                service_endpoint=self.service_endpoint,
                auth_file_location=self.auth_file_location,
                auth_profile=self.auth_profile,
                timeout=(10, 240),
                oci_module=self._oci,
            )
            self.client = self._oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )
        self._client_lock = threading.Lock()
        self.last_response_metadata = None

    # ------------------------------------------------------------------
    # Provider inference
    # ------------------------------------------------------------------

    def _infer_provider(self, model: str) -> str:
        if model.startswith(CUSTOM_ENDPOINT_PREFIX):
            return "generic"
        if model.startswith("cohere."):
            return "cohere"
        return "generic"

    def _is_openai_gpt5_family(self) -> bool:
        return self.model.lower().startswith("openai.gpt-5")

    def _build_serving_mode(self) -> Any:
        models = self._oci.generative_ai_inference.models
        if self.model.startswith(CUSTOM_ENDPOINT_PREFIX):
            return models.DedicatedServingMode(endpoint_id=self.model)
        return models.OnDemandServingMode(model_id=self.model)

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _normalize_messages(
        self, messages: str | list[LLMMessage]
    ) -> list[LLMMessage]:
        return self._format_messages(messages)

    def _coerce_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item["text"]))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _build_generic_content(self, content: Any) -> list[Any]:
        """Translate CrewAI message content into OCI generic content objects."""
        models = self._oci.generative_ai_inference.models
        if isinstance(content, str):
            return [models.TextContent(text=content or ".")]

        if not isinstance(content, list):
            return [models.TextContent(text=self._coerce_text(content) or ".")]

        processed: list[Any] = []
        for item in content:
            if isinstance(item, str):
                processed.append(models.TextContent(text=item or "."))
            elif isinstance(item, Mapping) and item.get("type") == "text":
                processed.append(
                    models.TextContent(text=str(item.get("text", "")) or ".")
                )
            else:
                processed.append(
                    models.TextContent(text=self._coerce_text(item) or ".")
                )
        return processed or [models.TextContent(text=".")]

    def _build_generic_messages(self, messages: list[LLMMessage]) -> list[Any]:
        """Map CrewAI conversation messages into OCI generic chat messages."""
        models = self._oci.generative_ai_inference.models
        role_map = {
            "user": models.UserMessage,
            "assistant": models.AssistantMessage,
            "system": models.SystemMessage,
        }
        oci_messages: list[Any] = []

        for message in messages:
            role = str(message.get("role", "user")).lower()
            message_cls = role_map.get(role)
            if message_cls is None:
                logging.debug("Skipping unsupported OCI message role: %s", role)
                continue
            oci_messages.append(
                message_cls(
                    content=self._build_generic_content(message.get("content", "")),
                )
            )

        return oci_messages

    def _build_cohere_chat_history(
        self, messages: list[LLMMessage]
    ) -> tuple[list[Any], str]:
        """Translate CrewAI messages into Cohere's split history + message shape."""
        models = self._oci.generative_ai_inference.models
        chat_history: list[Any] = []

        for message in messages[:-1]:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")

            if role in ("user", "system"):
                message_cls = (
                    models.CohereUserMessage
                    if role == "user"
                    else models.CohereSystemMessage
                )
                chat_history.append(message_cls(message=self._coerce_text(content)))
            elif role == "assistant":
                chat_history.append(
                    models.CohereChatBotMessage(
                        message=self._coerce_text(content) or " ",
                    )
                )

        last_message = messages[-1] if messages else {"role": "user", "content": ""}
        message_text = self._coerce_text(last_message.get("content", ""))
        return chat_history, message_text

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_chat_request(
        self,
        messages: list[LLMMessage],
    ) -> Any:
        """Build the provider-specific OCI chat request for the current model."""
        models = self._oci.generative_ai_inference.models

        if self.oci_provider == "cohere":
            chat_history, message_text = self._build_cohere_chat_history(messages)
            request_kwargs: dict[str, Any] = {
                "message": message_text,
                "chat_history": chat_history,
                "api_format": models.BaseChatRequest.API_FORMAT_COHERE,
            }
        else:
            request_kwargs = {
                "messages": self._build_generic_messages(messages),
                "api_format": models.BaseChatRequest.API_FORMAT_GENERIC,
            }

        if self.temperature is not None and not self._is_openai_gpt5_family():
            request_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            if self.oci_provider == "generic" and self.model.lower().startswith("openai."):
                request_kwargs["max_completion_tokens"] = self.max_tokens
            else:
                request_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            request_kwargs["top_k"] = self.top_k

        if self.stop and not self._is_openai_gpt5_family():
            stop_key = "stop_sequences" if self.oci_provider == "cohere" else "stop"
            request_kwargs[stop_key] = list(self.stop)

        if self.oci_provider == "cohere":
            return models.CohereChatRequest(**request_kwargs)
        return models.GenericChatRequest(**request_kwargs)

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def _extract_text(self, response: Any) -> str:
        chat_response = response.data.chat_response
        if self.oci_provider == "cohere":
            if getattr(chat_response, "text", None):
                return chat_response.text or ""
            message = getattr(chat_response, "message", None)
            if message is not None:
                content = getattr(message, "content", None) or []
                return "".join(
                    part.text for part in content if getattr(part, "text", None)
                )
            return ""

        choices = getattr(chat_response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None) or []
        return "".join(part.text for part in content if getattr(part, "text", None))

    def _extract_usage(self, response: Any) -> dict[str, int]:
        chat_response = response.data.chat_response
        usage = getattr(chat_response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    def _extract_response_metadata(self, response: Any) -> dict[str, Any]:
        chat_response = response.data.chat_response
        metadata: dict[str, Any] = {}

        finish_reason = getattr(chat_response, "finish_reason", None)
        if finish_reason is None:
            choices = getattr(chat_response, "choices", None) or []
            if choices:
                finish_reason = getattr(choices[0], "finish_reason", None)

        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason

        usage = self._extract_usage(response)
        if usage:
            metadata["usage"] = usage

        return metadata

    # ------------------------------------------------------------------
    # Call paths
    # ------------------------------------------------------------------

    def _finalize_text_response(
        self,
        *,
        content: str,
        messages: list[LLMMessage],
        from_task: Task | None,
        from_agent: Agent | None,
    ) -> str:
        content = self._apply_stop_words(content)
        content = self._invoke_after_llm_call_hooks(messages, content, from_agent)
        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
        )
        return content

    def _call_impl(
        self,
        *,
        messages: str | list[LLMMessage],
        from_task: Task | None,
        from_agent: Agent | None,
    ) -> str:
        normalized_messages = (
            messages if isinstance(messages, list) else self._normalize_messages(messages)
        )
        chat_request = self._build_chat_request(normalized_messages)
        chat_details = self._oci.generative_ai_inference.models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=self._build_serving_mode(),
            chat_request=chat_request,
        )
        response = self._chat(chat_details)
        usage = self._extract_usage(response)
        if usage:
            self._track_token_usage_internal(usage)
        self.last_response_metadata = self._extract_response_metadata(response) or None

        content = self._extract_text(response)
        return self._finalize_text_response(
            content=content,
            messages=normalized_messages,
            from_task=from_task,
            from_agent=from_agent,
        )

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel | list[dict[str, Any]]:
        normalized_messages = self._normalize_messages(messages)

        with llm_call_context():
            try:
                self._emit_call_started_event(
                    messages=normalized_messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                )

                if not self._invoke_before_llm_call_hooks(
                    normalized_messages, from_agent
                ):
                    raise ValueError("LLM call blocked by before_llm_call hook")

                return self._call_impl(
                    messages=normalized_messages,
                    from_task=from_task,
                    from_agent=from_agent,
                )
            except Exception as error:
                error_message = f"OCI Generative AI call failed: {error!s}"
                self._emit_call_failed_event(
                    error=error_message,
                    from_task=from_task,
                    from_agent=from_agent,
                )
                raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        return await asyncio.to_thread(
            self.call,
            messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    # ------------------------------------------------------------------
    # Client serialization
    # ------------------------------------------------------------------

    def _chat(self, chat_details: Any) -> Any:
        with self._client_lock:
            return self.client.chat(chat_details)

    # ------------------------------------------------------------------
    # Capability declarations
    # ------------------------------------------------------------------

    def get_context_window_size(self) -> int:
        model_lower = self.model.lower()
        if model_lower.startswith("google.gemini"):
            return 1048576
        if model_lower.startswith("openai."):
            return 200000
        if model_lower.startswith("cohere."):
            return 128000
        if model_lower.startswith("meta."):
            return 131072
        return 131072

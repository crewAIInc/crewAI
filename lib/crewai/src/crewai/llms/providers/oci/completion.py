from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import contextmanager
import inspect
import json
import logging
import os
import re
import threading
from typing import TYPE_CHECKING, Any, Literal, cast
import uuid

from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.utilities.oci import create_oci_client_kwargs, get_oci_module
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"
DEFAULT_OCI_REGION = "us-chicago-1"
_OCI_SCHEMA_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
_OCI_TOOL_RESULT_GUIDANCE = (
    "You have received tool results above. Respond to the user with a helpful, "
    "natural language answer that incorporates the tool results. Do not output "
    "raw JSON or tool call syntax. If you need additional information, you may "
    "call another tool."
)
_OCI_RESERVED_REQUEST_KWARGS = {
    "tool_choice",
    "parallel_tool_calls",
    "tool_result_guidance",
}


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
        stream: bool = False,
        oci_provider: str | None = None,
        max_sequential_tool_calls: int = 8,
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
        self.stream = stream
        self.oci_provider = oci_provider or self._infer_provider(model)
        self.max_sequential_tool_calls = max_sequential_tool_calls
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
        self._client_condition = threading.Condition()
        self._next_client_ticket = 0
        self._active_client_ticket = 0
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
        return self.model.startswith("openai.gpt-5")

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

    def _message_has_multimodal_content(self, content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for item in content:
            if isinstance(item, Mapping) and item.get("type") not in (None, "text"):
                return True
        return False

    def _build_generic_content(self, content: Any) -> list[Any]:
        """Translate CrewAI message content into OCI generic content objects.

        Handles text, image_url, document_url, video_url, and audio_url types.
        """
        models = self._oci.generative_ai_inference.models
        if isinstance(content, str):
            return [models.TextContent(text=content or ".")]

        if not isinstance(content, list):
            return [models.TextContent(text=self._coerce_text(content) or ".")]

        processed: list[Any] = []
        for item in content:
            if isinstance(item, str):
                processed.append(models.TextContent(text=item))
                continue
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"OCI message content items must be strings or dictionaries, got: {type(item)}"
                )

            content_type = item.get("type")
            if content_type == "text":
                processed.append(
                    models.TextContent(text=str(item.get("text", "")) or ".")
                )
            elif content_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, Mapping) else None
                if not url:
                    raise ValueError("OCI image_url content requires image_url.url")
                processed.append(
                    models.ImageContent(image_url=models.ImageUrl(url=url))
                )
            elif content_type in ("document_url", "document", "file"):
                doc_data = item.get("document_url") or item.get("document") or item.get("file")
                url = doc_data.get("url") if isinstance(doc_data, Mapping) else item.get("url")
                if not url:
                    raise ValueError("OCI document content requires a url")
                processed.append(
                    models.DocumentContent(document_url=models.DocumentUrl(url=url))
                )
            elif content_type in ("video_url", "video"):
                video_data = item.get("video_url") or item.get("video")
                url = video_data.get("url") if isinstance(video_data, Mapping) else item.get("url")
                if not url:
                    raise ValueError("OCI video content requires a url")
                processed.append(
                    models.VideoContent(video_url=models.VideoUrl(url=url))
                )
            elif content_type in ("audio_url", "audio"):
                audio_data = item.get("audio_url") or item.get("audio")
                url = audio_data.get("url") if isinstance(audio_data, Mapping) else item.get("url")
                if not url:
                    raise ValueError("OCI audio content requires a url")
                processed.append(
                    models.AudioContent(audio_url=models.AudioUrl(url=url))
                )
            else:
                raise ValueError(f"Unsupported OCI content type: {content_type}")

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
            if role == "tool":
                tool_kwargs: dict[str, Any] = {
                    "content": self._build_generic_content(message.get("content", "")),
                }
                if message.get("tool_call_id"):
                    tool_kwargs["tool_call_id"] = message["tool_call_id"]
                oci_messages.append(models.ToolMessage(**tool_kwargs))
                continue

            message_cls = role_map.get(role)
            if message_cls is None:
                logging.debug("Skipping unsupported OCI message role: %s", role)
                continue

            message_kwargs: dict[str, Any] = {
                "content": self._build_generic_content(message.get("content", "")),
            }
            if role == "assistant" and message.get("tool_calls"):
                message_kwargs["tool_calls"] = [
                    models.FunctionCall(
                        id=tool_call.get("id"),
                        name=tool_call.get("function", {}).get("name"),
                        arguments=tool_call.get("function", {}).get("arguments", "{}"),
                    )
                    for tool_call in message.get("tool_calls", [])
                    if tool_call.get("function", {}).get("name")
                ]
                if not message_kwargs["content"]:
                    message_kwargs["content"] = [models.TextContent(text=".")]

            oci_messages.append(message_cls(**message_kwargs))

        if (
            self._tool_result_guidance_enabled()
            and any(str(message.get("role", "")).lower() == "tool" for message in messages)
        ):
            oci_messages.append(
                models.SystemMessage(
                    content=[models.TextContent(text=_OCI_TOOL_RESULT_GUIDANCE)]
                )
            )

        return oci_messages

    def _build_cohere_chat_history(
        self, messages: list[LLMMessage]
    ) -> tuple[list[Any], list[Any] | None, str]:
        """Translate CrewAI messages into Cohere's split history/tool-results shape."""
        models = self._oci.generative_ai_inference.models
        chat_history: list[Any] = []
        trailing_tool_count = 0
        for message in reversed(messages):
            if str(message.get("role", "")).lower() != "tool":
                break
            trailing_tool_count += 1

        history_messages = (
            messages[:-trailing_tool_count] if trailing_tool_count else messages[:-1]
        )

        for message in history_messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")
            if self._message_has_multimodal_content(content):
                raise ValueError(
                    "OCI Cohere models currently support text-only messages in CrewAI."
                )

            if role in ("user", "system"):
                message_cls = (
                    models.CohereUserMessage
                    if role == "user"
                    else models.CohereSystemMessage
                )
                chat_history.append(message_cls(message=self._coerce_text(content)))
            elif role == "assistant":
                tool_calls = None
                if message.get("tool_calls"):
                    tool_calls = []
                    for tool_call in message.get("tool_calls", []):
                        function_info = tool_call.get("function", {})
                        function_name = function_info.get("name")
                        if not function_name:
                            continue
                        raw_arguments = function_info.get("arguments", "{}")
                        if isinstance(raw_arguments, str):
                            try:
                                parameters = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                parameters = {}
                        elif isinstance(raw_arguments, Mapping):
                            parameters = dict(raw_arguments)
                        else:
                            parameters = {}
                        tool_calls.append(
                            models.CohereToolCall(name=function_name, parameters=parameters)
                        )
                chat_history.append(
                    models.CohereChatBotMessage(
                        message=self._coerce_text(content) or " ",
                        tool_calls=tool_calls,
                    )
                )
            elif role == "tool":
                tool_name = message.get("name") or "tool"
                chat_history.append(
                    models.CohereToolMessage(
                        tool_results=[
                            models.CohereToolResult(
                                call=models.CohereToolCall(name=tool_name, parameters={}),
                                outputs=[{"output": self._coerce_text(content)}],
                            )
                        ]
                    )
                )

        last_message = messages[-1] if messages else {"role": "user", "content": ""}
        tool_results: list[Any] = []
        if str(last_message.get("role", "user")).lower() == "tool":
            previous_tool_calls: dict[str, dict[str, Any]] = {}
            for message in messages:
                if str(message.get("role", "")).lower() != "assistant":
                    continue
                for tool_call in message.get("tool_calls", []):
                    tool_call_id = tool_call.get("id")
                    if not tool_call_id:
                        continue
                    function_info = tool_call.get("function", {})
                    raw_arguments = function_info.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            parameters = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            parameters = {}
                    elif isinstance(raw_arguments, Mapping):
                        parameters = dict(raw_arguments)
                    else:
                        parameters = {}
                    previous_tool_calls[tool_call_id] = {
                        "name": function_info.get("name", "tool"),
                        "parameters": parameters,
                    }

            for message in messages[-trailing_tool_count:]:
                if str(message.get("role", "")).lower() != "tool":
                    continue
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    continue
                previous_call = previous_tool_calls.get(tool_call_id, {})
                tool_results.append(
                    models.CohereToolResult(
                        call=models.CohereToolCall(
                            name=previous_call.get("name", message.get("name", "tool")),
                            parameters=previous_call.get("parameters", {}),
                        ),
                        outputs=[{"output": self._coerce_text(message.get("content", ""))}],
                    )
                )

        message_text = self._coerce_text(last_message.get("content", ""))
        if tool_results:
            message_text = ""

        return chat_history, tool_results or None, message_text

    # ------------------------------------------------------------------
    # Tool formatting
    # ------------------------------------------------------------------

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[Any]:
        if not tools:
            return []
        models = self._oci.generative_ai_inference.models
        formatted: list[Any] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function_spec = tool.get("function", {})
            if not isinstance(function_spec, Mapping):
                continue
            name = function_spec.get("name")
            if not name:
                continue
            parameters = function_spec.get("parameters", {})
            if not isinstance(parameters, Mapping):
                parameters = {}

            if self.oci_provider == "cohere":
                param_defs = {}
                required = set(parameters.get("required", []))
                for pname, pschema in parameters.get("properties", {}).items():
                    if not isinstance(pschema, Mapping):
                        continue
                    param_defs[pname] = models.CohereParameterDefinition(
                        description=pschema.get("description", ""),
                        type=pschema.get("type", "object"),
                        is_required=pname in required,
                    )
                formatted.append(models.CohereTool(
                    name=name,
                    description=function_spec.get("description", name),
                    parameter_definitions=param_defs,
                ))
            else:
                formatted.append(models.FunctionDefinition(
                    name=name,
                    description=function_spec.get("description", name),
                    parameters={
                        "type": parameters.get("type", "object"),
                        "properties": parameters.get("properties", {}),
                        "required": parameters.get("required", []),
                    },
                ))
        return formatted

    def _tool_result_guidance_enabled(self) -> bool:
        return bool(self.additional_params.get("tool_result_guidance"))

    def _parallel_tool_calls_enabled(self) -> bool:
        return bool(self.additional_params.get("parallel_tool_calls"))

    def _build_tool_choice(self) -> Any | None:
        tool_choice = self.additional_params.get("tool_choice")
        if tool_choice is None:
            return None
        models = self._oci.generative_ai_inference.models
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return models.ToolChoiceAuto()
            if tool_choice == "none":
                return models.ToolChoiceNone()
            if tool_choice in ("any", "required"):
                return models.ToolChoiceRequired()
            return models.ToolChoiceFunction(name=tool_choice)
        if isinstance(tool_choice, bool):
            return models.ToolChoiceRequired() if tool_choice else models.ToolChoiceNone()
        if isinstance(tool_choice, Mapping):
            fn = tool_choice.get("function")
            if isinstance(fn, Mapping) and fn.get("name"):
                return models.ToolChoiceFunction(name=str(fn["name"]))
            return models.ToolChoiceAuto()
        raise ValueError("Unrecognized OCI tool_choice. Expected str, bool, or function mapping.")

    def _allowed_passthrough_request_keys(self, request_cls: type[Any]) -> set[str]:
        """Return request attributes that can safely be forwarded to the OCI SDK."""
        attribute_map = getattr(request_cls, "attribute_map", None)
        if isinstance(attribute_map, Mapping):
            return {str(key) for key in attribute_map}
        swagger_types = getattr(request_cls, "swagger_types", None)
        if isinstance(swagger_types, Mapping):
            return {str(key) for key in swagger_types}
        signature = inspect.signature(request_cls)
        return {
            name
            for name, param in signature.parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_response_format(
        self, response_model: type[BaseModel] | None
    ) -> Any | None:
        if response_model is None:
            return None
        models = self._oci.generative_ai_inference.models
        schema_description = generate_model_description(response_model)["json_schema"]
        schema_name = _OCI_SCHEMA_NAME_PATTERN.sub("_", schema_description["name"])
        json_schema = models.ResponseJsonSchema(
            name=schema_name,
            description=(response_model.__doc__ or "").strip() or schema_name,
            schema=schema_description["schema"],
            is_strict=schema_description["strict"],
        )
        if self.oci_provider == "cohere":
            return models.CohereResponseJsonFormat(schema=json_schema.schema)
        return models.JsonSchemaResponseFormat(json_schema=json_schema)

    def _build_chat_request(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        *,
        is_stream: bool = False,
    ) -> Any:
        """Build the provider-specific OCI chat request for the current model."""
        models = self._oci.generative_ai_inference.models

        if self.oci_provider == "cohere":
            if any(self._message_has_multimodal_content(msg.get("content")) for msg in messages):
                raise ValueError(
                    "OCI Cohere models currently support text-only messages in CrewAI."
                )
            chat_history, tool_results, message_text = self._build_cohere_chat_history(
                messages
            )
            request_kwargs: dict[str, Any] = {
                "message": message_text,
                "chat_history": chat_history,
                "api_format": models.BaseChatRequest.API_FORMAT_COHERE,
            }
            if tool_results:
                request_kwargs["tool_results"] = tool_results
        else:
            request_kwargs = {
                "messages": self._build_generic_messages(messages),
                "api_format": models.BaseChatRequest.API_FORMAT_GENERIC,
            }

        if self.temperature is not None and not self._is_openai_gpt5_family():
            request_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            if self.oci_provider == "generic" and self.model.startswith("openai."):
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

        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            request_kwargs["tools"] = formatted_tools
            if self.oci_provider == "cohere":
                if self._parallel_tool_calls_enabled():
                    raise ValueError("OCI Cohere models do not support parallel_tool_calls.")
                request_kwargs.setdefault("is_force_single_step", False)
            else:
                tool_choice = self._build_tool_choice()
                if tool_choice is not None:
                    request_kwargs["tool_choice"] = tool_choice
                if self._parallel_tool_calls_enabled():
                    request_kwargs["is_parallel_tool_calls"] = True

        response_format = self._build_response_format(response_model)
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        if is_stream:
            request_kwargs["is_stream"] = True
            request_kwargs["stream_options"] = models.StreamOptions(
                is_include_usage=True
            )

        if self.oci_provider == "cohere":
            allowed = self._allowed_passthrough_request_keys(models.CohereChatRequest)
            passthrough = {
                k: v for k, v in self.additional_params.items()
                if k not in _OCI_RESERVED_REQUEST_KWARGS and k in allowed
            }
            request_kwargs.update(passthrough)
            return models.CohereChatRequest(**request_kwargs)

        allowed = self._allowed_passthrough_request_keys(models.GenericChatRequest)
        passthrough = {
            k: v for k, v in self.additional_params.items()
            if k not in _OCI_RESERVED_REQUEST_KWARGS and k in allowed
        }
        request_kwargs.update(passthrough)
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

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        """Normalize provider-specific tool calls back into CrewAI's shape."""
        chat_response = response.data.chat_response
        raw: list[Any] = []
        if self.oci_provider == "cohere":
            raw = getattr(chat_response, "tool_calls", None) or []
        else:
            choices = getattr(chat_response, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                raw = getattr(msg, "tool_calls", None) or []

        if self.oci_provider == "cohere":
            return [
                {
                    "id": uuid.uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": getattr(tc, "name", ""),
                        "arguments": json.dumps(getattr(tc, "parameters", {}) or {}),
                    },
                }
                for tc in raw
            ]
        return [
            {
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {
                    "name": getattr(tc, "name", ""),
                    "arguments": getattr(tc, "arguments", "{}"),
                },
            }
            for tc in raw
        ]

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
    # Streaming extraction
    # ------------------------------------------------------------------

    def _parse_stream_event(self, event: Any) -> dict[str, Any]:
        """Convert OCI SSE event payloads into plain dicts."""
        event_data = getattr(event, "data", None)
        if not event_data:
            return {}
        if isinstance(event_data, str):
            try:
                parsed = json.loads(event_data)
                if isinstance(parsed, Mapping):
                    return dict(parsed)
                return {}
            except json.JSONDecodeError:
                logging.debug("Skipping invalid OCI SSE payload: %s", event_data)
                return {}
        if isinstance(event_data, Mapping):
            return dict(event_data)
        return {}

    def _extract_text_from_stream_event(self, event_data: dict[str, Any]) -> str:
        if self.oci_provider == "cohere":
            if "text" in event_data:
                return str(event_data.get("text", ""))
            message = event_data.get("message", {})
            if isinstance(message, Mapping):
                content = message.get("content", [])
                if isinstance(content, list):
                    return "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, Mapping)
                    )
            return ""

        message = event_data.get("message", {})
        if not isinstance(message, Mapping):
            return ""
        content = message.get("content", [])
        if not isinstance(content, list):
            return ""
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("text")
        )

    def _extract_tool_calls_from_stream_event(
        self, event_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        message = event_data.get("message", {})
        if self.oci_provider == "cohere":
            raw = event_data.get("toolCalls", [])
        else:
            raw = message.get("toolCalls", []) if isinstance(message, Mapping) else []
        if not isinstance(raw, list):
            return []
        if self.oci_provider == "cohere":
            return [
                {"id": None, "type": "function", "function": {
                    "name": str(tc.get("name", "")),
                    "arguments": json.dumps(tc.get("parameters", {})),
                }}
                for tc in raw if isinstance(tc, Mapping)
            ]
        return [
            {"id": tc.get("id"), "type": "function", "function": {
                "name": tc.get("name"), "arguments": tc.get("arguments"),
            }}
            for tc in raw if isinstance(tc, Mapping)
        ]

    def _extract_usage_from_stream_event(self, event_data: dict[str, Any]) -> dict[str, int]:
        usage = event_data.get("usage")
        if not isinstance(usage, Mapping):
            return {}
        return {
            "prompt_tokens": int(usage.get("promptTokens", 0) or 0),
            "completion_tokens": int(usage.get("completionTokens", 0) or 0),
            "total_tokens": int(usage.get("totalTokens", 0) or 0),
        }

    def _extract_metadata_from_stream_event(self, event_data: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        finish_reason = event_data.get("finishReason")
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        usage = self._extract_usage_from_stream_event(event_data)
        if usage:
            metadata["usage"] = usage
        return metadata

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    def _parse_structured_response(
        self,
        *,
        content: str,
        response_model: type[BaseModel],
        messages: list[LLMMessage],
        from_task: Task | None,
        from_agent: Agent | None,
    ) -> BaseModel:
        try:
            structured_response = self._validate_structured_output(
                content, response_model
            )
        except Exception as error:
            raise ValueError(
                f"Failed to validate OCI structured response with model "
                f"{response_model.__name__}: {error}"
            ) from error

        if not isinstance(structured_response, BaseModel):
            raise ValueError(
                f"OCI structured response parsing returned unexpected type: "
                f"{type(structured_response)}"
            )

        self._emit_call_completed_event(
            response=structured_response.model_dump_json(),
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=messages,
        )
        return structured_response

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _handle_tool_calls(
        self,
        *,
        normalized_messages: list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None,
        callbacks: list[Any] | None,
        available_functions: dict[str, Any] | None,
        from_task: Task | None,
        from_agent: Agent | None,
        tool_depth: int,
        response_model: type[BaseModel] | None = None,
        tool_calls: list[dict[str, Any]],
    ) -> str | BaseModel | list[dict[str, Any]]:
        """Execute one round of tool calls and recurse until the model finishes."""
        if tool_calls and not available_functions:
            self._emit_call_completed_event(
                response=tool_calls,
                call_type=LLMCallType.TOOL_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=normalized_messages,
            )
            return tool_calls

        if tool_depth >= self.max_sequential_tool_calls:
            raise RuntimeError(
                "OCI native provider exceeded max_sequential_tool_calls."
            )

        next_messages = list(normalized_messages)
        next_messages.append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        )

        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    fn_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    fn_args = {}
            elif isinstance(raw_args, Mapping):
                fn_args = dict(raw_args)
            else:
                fn_args = {}

            result = self._handle_tool_execution(
                function_name=fn_name,
                function_args=fn_args,
                available_functions=available_functions or {},
                from_task=from_task,
                from_agent=from_agent,
            )
            if result is None:
                result = f"Tool '{fn_name}' failed or returned no result."

            next_messages.append({
                "role": "tool",
                "tool_call_id": str(tool_call.get("id") or uuid.uuid4().hex),
                "name": fn_name,
                "content": str(result),
            })

        if self.stream:
            return self._stream_call_impl(
                messages=next_messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                tool_depth=tool_depth + 1,
                response_model=response_model,
            )
        return self._call_impl(
            messages=next_messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            tool_depth=tool_depth + 1,
            response_model=response_model,
        )

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
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        tool_depth: int = 0,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel | list[dict[str, Any]]:
        normalized_messages = (
            messages if isinstance(messages, list) else self._normalize_messages(messages)
        )
        chat_request = self._build_chat_request(
            normalized_messages, tools=tools, response_model=response_model
        )
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
        tool_calls = self._extract_tool_calls(response)
        if tool_calls:
            return self._handle_tool_calls(
                normalized_messages=normalized_messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                tool_depth=tool_depth,
                response_model=response_model,
                tool_calls=tool_calls,
            )

        if response_model is not None:
            return self._parse_structured_response(
                content=content,
                response_model=response_model,
                messages=normalized_messages,
                from_task=from_task,
                from_agent=from_agent,
            )

        return self._finalize_text_response(
            content=content,
            messages=normalized_messages,
            from_task=from_task,
            from_agent=from_agent,
        )

    def _stream_call_impl(
        self,
        *,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        tool_depth: int = 0,
        response_model: type[BaseModel] | None = None,
    ) -> str | BaseModel | list[dict[str, Any]]:
        """Handle OCI streaming while reconstructing final text/tool state."""
        normalized_messages = (
            messages if isinstance(messages, list) else self._normalize_messages(messages)
        )
        chat_request = self._build_chat_request(
            normalized_messages, tools=tools, response_model=response_model,
            is_stream=True,
        )
        chat_details = self._oci.generative_ai_inference.models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=self._build_serving_mode(),
            chat_request=chat_request,
        )
        full_response = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage_data: dict[str, int] = {}
        response_metadata: dict[str, Any] = {}
        response_id = uuid.uuid4().hex

        for event in self._stream_chat_events(chat_details):
            event_data = self._parse_stream_event(event)
            if not event_data:
                continue

            text_chunk = self._extract_text_from_stream_event(event_data)
            if text_chunk:
                full_response += text_chunk
                self._emit_stream_chunk_event(
                    chunk=text_chunk,
                    from_task=from_task,
                    from_agent=from_agent,
                    call_type=LLMCallType.LLM_CALL,
                    response_id=response_id,
                )

            stream_tool_calls = self._extract_tool_calls_from_stream_event(event_data)
            for index, tc in enumerate(stream_tool_calls):
                state = tool_calls_by_index.setdefault(
                    index,
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                if tc.get("id"):
                    state["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    state["function"]["name"] = fn["name"]
                chunk_args = fn.get("arguments")
                if chunk_args:
                    state["function"]["arguments"] += str(chunk_args)
                self._emit_stream_chunk_event(
                    chunk=str(chunk_args or ""),
                    tool_call={"id": state["id"], "type": "function", "function": {
                        "name": state["function"]["name"],
                        "arguments": str(chunk_args or ""),
                    }},
                    from_task=from_task,
                    from_agent=from_agent,
                    call_type=LLMCallType.TOOL_CALL,
                    response_id=response_id,
                )

            usage_chunk = self._extract_usage_from_stream_event(event_data)
            if usage_chunk:
                usage_data = usage_chunk
            response_metadata.update(self._extract_metadata_from_stream_event(event_data))

        if usage_data:
            self._track_token_usage_internal(usage_data)
            response_metadata["usage"] = usage_data
        self.last_response_metadata = response_metadata or None

        tool_calls = [
            {
                "id": tc.get("id") or uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": tc["function"].get("name", "") or "",
                    "arguments": tc["function"].get("arguments", "") or "",
                },
            }
            for _, tc in sorted(tool_calls_by_index.items())
            if tc["function"].get("name")
        ]

        if tool_calls:
            return self._handle_tool_calls(
                normalized_messages=normalized_messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                tool_depth=tool_depth,
                response_model=response_model,
                tool_calls=tool_calls,
            )

        if response_model is not None:
            return self._parse_structured_response(
                content=full_response,
                response_model=response_model,
                messages=normalized_messages,
                from_task=from_task,
                from_agent=from_agent,
            )

        return self._finalize_text_response(
            content=full_response,
            messages=normalized_messages,
            from_task=from_task,
            from_agent=from_agent,
        )

    def iter_stream(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> Any:
        """Yield raw text chunks from OCI without triggering tool recursion."""
        normalized_messages = self._normalize_messages(messages)
        chat_request = self._build_chat_request(normalized_messages, is_stream=True)
        chat_details = self._oci.generative_ai_inference.models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=self._build_serving_mode(),
            chat_request=chat_request,
        )
        response = self._chat(chat_details)
        usage_data: dict[str, int] = {}
        response_metadata: dict[str, Any] = {}

        for event in response.data.events():
            event_data = self._parse_stream_event(event)
            if not event_data:
                continue
            text_chunk = self._extract_text_from_stream_event(event_data)
            if text_chunk:
                yield text_chunk
            usage_chunk = self._extract_usage_from_stream_event(event_data)
            if usage_chunk:
                usage_data = usage_chunk
            response_metadata.update(self._extract_metadata_from_stream_event(event_data))

        if usage_data:
            self._track_token_usage_internal(usage_data)
            response_metadata["usage"] = usage_data
        self.last_response_metadata = response_metadata or None

    async def astream(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
    ) -> Any:
        """Async streaming — true async via aiohttp when available, thread fallback otherwise."""
        if self._async_client is None:
            # Fallback: sync stream bridged via thread
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            error_holder: list[BaseException] = []

            def _producer() -> None:
                try:
                    for chunk in self.iter_stream(
                        messages=messages, tools=tools, callbacks=callbacks,
                        available_functions=available_functions, from_task=from_task,
                        from_agent=from_agent,
                    ):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except BaseException as error:
                    error_holder.append(error)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            thread = threading.Thread(target=_producer, daemon=True)
            thread.start()
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            thread.join()
            if error_holder:
                raise error_holder[0]
            return

        normalized_messages = self._normalize_messages(messages)
        request_data = self._prepare_async_request(
            normalized_messages, tools=tools, is_stream=True
        )
        usage_data: dict[str, int] = {}
        response_metadata: dict[str, Any] = {}

        async for event_data in self._async_client.chat_async(
            compartment_id=request_data["compartment_id"],
            chat_request_dict=request_data["chat_request_dict"],
            serving_mode_dict=request_data["serving_mode_dict"],
            stream=True,
        ):
            text_chunk = self._extract_text_from_stream_event(event_data)
            if text_chunk:
                yield text_chunk

            usage_chunk = self._extract_usage_from_stream_event(event_data)
            if usage_chunk:
                usage_data = usage_chunk
            response_metadata.update(self._extract_metadata_from_stream_event(event_data))

        if usage_data:
            self._track_token_usage_internal(usage_data)
            response_metadata["usage"] = usage_data
        self.last_response_metadata = response_metadata or None

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

                if self.stream:
                    return self._stream_call_impl(
                        messages=normalized_messages,
                        tools=tools,
                        callbacks=callbacks,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                        tool_depth=0,
                        response_model=response_model,
                    )

                return self._call_impl(
                    messages=normalized_messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    tool_depth=0,
                    response_model=response_model,
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
        """Async call — true async via aiohttp when available, thread fallback otherwise."""
        if self._async_client is None:
            return await asyncio.to_thread(
                self.call, messages, tools=tools, callbacks=callbacks,
                available_functions=available_functions, from_task=from_task,
                from_agent=from_agent, response_model=response_model,
            )

        normalized_messages = self._normalize_messages(messages)
        request_data = self._prepare_async_request(
            normalized_messages, tools=tools, response_model=response_model
        )

        response_data = None
        async for data in self._async_client.chat_async(
            compartment_id=request_data["compartment_id"],
            chat_request_dict=request_data["chat_request_dict"],
            serving_mode_dict=request_data["serving_mode_dict"],
            stream=False,
        ):
            response_data = data
            break

        if response_data is None:
            raise RuntimeError("No response received from OCI GenAI async call")

        # Extract text from the raw JSON response
        chat_response = response_data.get("chatResponse", {})
        content = self._extract_text_from_async_response(chat_response)
        content = self._apply_stop_words(content)

        # Track usage
        usage = chat_response.get("usage", {})
        if usage:
            usage_dict = {
                "prompt_tokens": usage.get("promptTokens", 0),
                "completion_tokens": usage.get("completionTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
            self._track_token_usage_internal(usage_dict)

        self._emit_call_completed_event(
            response=content,
            call_type=LLMCallType.LLM_CALL,
            from_task=from_task,
            from_agent=from_agent,
            messages=normalized_messages,
        )
        return content

    def _extract_text_from_async_response(self, chat_response: dict[str, Any]) -> str:
        """Extract text from a raw JSON chat response dict."""
        # Generic format: choices[0].message.content[].text
        choices = chat_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", [])
            if isinstance(content, list):
                return "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                )
            return str(content) if content else ""

        # Cohere format: text field or message.content
        text = chat_response.get("text")
        if text:
            return str(text)

        message = chat_response.get("message", {})
        content = message.get("content", [])
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        return ""

    # ------------------------------------------------------------------
    # Async support (true async via aiohttp, no thread wrappers)
    # ------------------------------------------------------------------

    @property
    def _async_client(self) -> Any | None:
        """Lazy-init true async client reusing the sync client's signer.

        Returns None if aiohttp is not installed or the client can't be created,
        in which case callers fall back to asyncio.to_thread.
        """
        if not hasattr(self, "_async_client_instance"):
            self._async_client_instance = None
            try:
                from crewai.utilities.oci_async import OCIAsyncClient

                oci = self._oci
                expected_cls = oci.generative_ai_inference.GenerativeAiInferenceClient
                if isinstance(self.client, expected_cls):
                    base_client = self.client.base_client
                    self._async_client_instance = OCIAsyncClient(
                        service_endpoint=self.service_endpoint,
                        signer=base_client.signer,
                        config=getattr(base_client, "config", {}),
                    )
            except (ImportError, Exception):
                pass
        return self._async_client_instance

    def _prepare_async_request(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        *,
        is_stream: bool = False,
    ) -> dict[str, Any]:
        """Build request dicts for the async client (SDK objects → JSON dicts)."""
        from oci.util import to_dict

        chat_request = self._build_chat_request(
            messages, tools=tools, response_model=response_model, is_stream=is_stream
        )
        chat_details = self._oci.generative_ai_inference.models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=self._build_serving_mode(),
            chat_request=chat_request,
        )
        return {
            "compartment_id": chat_details.compartment_id,
            "chat_request_dict": to_dict(chat_details.chat_request),
            "serving_mode_dict": to_dict(chat_details.serving_mode),
        }

    # ------------------------------------------------------------------
    # Client serialization (sync)
    # ------------------------------------------------------------------

    def _chat(self, chat_details: Any) -> Any:
        with self._ordered_client_access():
            return self.client.chat(chat_details)

    def _stream_chat_events(self, chat_details: Any) -> Any:
        """Yield streaming events while holding the shared OCI client lock."""
        with self._ordered_client_access():
            response = self.client.chat(chat_details)
            yield from response.data.events()

    @contextmanager
    def _ordered_client_access(self) -> Any:
        """Serialize shared OCI client access in call-arrival order."""
        with self._client_condition:
            ticket = self._next_client_ticket
            self._next_client_ticket += 1
            while ticket != self._active_client_ticket:
                self._client_condition.wait()

        try:
            yield
        finally:
            with self._client_condition:
                self._active_client_ticket += 1
                self._client_condition.notify_all()

    # ------------------------------------------------------------------
    # Capability declarations
    # ------------------------------------------------------------------

    def supports_function_calling(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return True

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

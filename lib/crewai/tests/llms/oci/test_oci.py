from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.llms.providers.oci.completion import OCICompletion


def test_oci_completion_is_used_when_oci_provider(
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value.chat.return_value = oci_response_factories[
        "chat"
    ]("test")

    llm = LLM(
        model=str(oci_unit_values["prefixed_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    assert isinstance(llm, OCICompletion)
    assert llm.provider == "oci"
    assert llm.model == str(oci_unit_values["generic_model"])


def test_oci_completion_infers_provider_family(
    patch_oci_module,
    oci_response_factories,
    oci_provider_family_case: tuple[str, str],
    oci_unit_values: dict[str, object],
):
    model, oci_provider = oci_provider_family_case
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value.chat.return_value = oci_response_factories[
        "chat"
    ]("test")

    llm = OCICompletion(
        model=model,
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    assert llm.oci_provider == oci_provider


def test_oci_completion_initialization_parameters(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = LLM(
        model=str(oci_unit_values["prefixed_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        service_endpoint=str(oci_unit_values["service_endpoint"]),
        temperature=0.2,
        max_tokens=256,
        top_p=0.9,
        top_k=20,
    )

    assert isinstance(llm, OCICompletion)
    assert llm.temperature == 0.2
    assert llm.max_tokens == 256
    assert llm.top_p == 0.9
    assert llm.top_k == 20
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.assert_called_once()


def test_oci_completion_call_uses_chat_api(
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]("Hello from OCI")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["chat_prompt"])}]
    )

    assert result == "Hello from OCI"
    request = fake_client.chat.call_args.args[0]
    assert request.compartment_id == str(oci_unit_values["compartment_id"])
    assert request.serving_mode.model_id == str(oci_unit_values["generic_model"])
    assert request.chat_request.messages[0].content[0].text == str(
        oci_unit_values["chat_prompt"]
    )


def test_oci_completion_treats_none_content_as_empty_text(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    content = llm._build_generic_content(None)

    assert content[0].text == "."


def test_oci_completion_call_normalizes_messages_once(
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]("Hello from OCI")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    normalize_call_count = 0
    original_normalize_messages = llm._normalize_messages

    def counting_normalize(messages):
        nonlocal normalize_call_count
        normalize_call_count += 1
        return original_normalize_messages(messages)

    monkeypatch.setattr(llm, "_normalize_messages", counting_normalize)

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["chat_prompt"])}]
    )

    assert result == "Hello from OCI"
    assert normalize_call_count == 1


def test_oci_completion_uses_region_to_build_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value.chat.return_value = oci_response_factories[
        "chat"
    ]("test")
    monkeypatch.setenv("OCI_REGION", str(oci_unit_values["region"]))

    llm = OCICompletion(
        model=str(oci_unit_values["generic_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    assert (
        llm.service_endpoint
        == f"https://inference.generativeai.{oci_unit_values['region']}.oci.oraclecloud.com"
    )


def test_oci_openai_models_use_max_completion_tokens(
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value.chat.return_value = oci_response_factories[
        "chat"
    ]("test")

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        max_tokens=77,
    )

    request = llm._build_chat_request([{"role": "user", "content": "hello"}])

    assert not hasattr(request, "max_tokens")
    assert request.max_completion_tokens == 77


def test_oci_openai_gpt5_omits_unsupported_temperature_and_stop(
    patch_oci_module,
    oci_response_factories,
    oci_unit_values: dict[str, object],
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value.chat.return_value = oci_response_factories[
        "chat"
    ]("test")

    llm = OCICompletion(
        model=str(oci_unit_values["gpt5_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        temperature=0,
        stop=["Observation:"],
    )

    request = llm._build_chat_request([{"role": "user", "content": "hello"}])

    assert not hasattr(request, "temperature")
    assert not hasattr(request, "stop")


def test_oci_completion_supports_structured_output(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    class OracleCloudSummary(BaseModel):
        summary: str
        confidence: int

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"](
        '{"summary":"OCI is scalable","confidence":92}'
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_structured_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["json_prompt"])}],
        response_model=OracleCloudSummary,
    )

    assert isinstance(result, OracleCloudSummary)
    assert result.summary == "OCI is scalable"
    assert result.confidence == 92
    request = fake_client.chat.call_args.args[0]
    assert request.chat_request.response_format.json_schema.name == "OracleCloudSummary"
    assert request.chat_request.response_format.json_schema.is_strict is True


def test_oci_completion_extracts_fenced_structured_output(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    class OracleCloudSummary(BaseModel):
        summary: str
        topic: str

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"](
        '```json\n{"summary":"OCI is scalable","topic":"Oracle Cloud"}\n```'
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["gpt4_control_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["json_prompt"])}],
        response_model=OracleCloudSummary,
    )

    assert isinstance(result, OracleCloudSummary)
    assert result.summary == "OCI is scalable"
    assert result.topic == "Oracle Cloud"


def test_oci_completion_streams_generic_responses(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["stream"](
        {"message": {"content": [{"text": "Hello"}]}},
        {"message": {"content": [{"text": " from OCI"}]}},
        {"finishReason": "stop"},
        {"usage": {"promptTokens": 11, "completionTokens": 4, "totalTokens": 15}},
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        stream=True,
    )
    llm._emit_stream_chunk_event = MagicMock()

    result = llm.call([{"role": "user", "content": str(oci_unit_values["hello_prompt"])}])

    assert result == "Hello from OCI"
    request = fake_client.chat.call_args.args[0]
    assert request.chat_request.is_stream is True
    assert request.chat_request.stream_options.is_include_usage is True
    assert llm._emit_stream_chunk_event.call_count == 2


def test_oci_completion_builds_multimodal_generic_messages(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    request = llm._build_chat_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(oci_unit_values["multimodal_prompt"])},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                    {
                        "type": "document_url",
                        "document_url": {"url": "data:application/pdf;base64,BBB"},
                    },
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,CCC"}},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,DDD"}},
                ],
            }
        ]
    )

    content = request.messages[0].content
    assert content[0].text == str(oci_unit_values["multimodal_prompt"])
    assert content[1].image_url.url == "data:image/png;base64,AAA"
    assert content[2].document_url.url == "data:application/pdf;base64,BBB"
    assert content[3].video_url.url == "data:video/mp4;base64,CCC"
    assert content[4].audio_url.url == "data:audio/wav;base64,DDD"


def test_oci_cohere_models_use_cohere_request_format(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["cohere_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    request = llm._build_chat_request(
        [{"role": "user", "content": str(oci_unit_values["search_prompt"])}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search documentation",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )

    assert request.api_format == "COHERE"
    assert request.message == str(oci_unit_values["search_prompt"])
    assert request.tools[0].name == "search_docs"
    assert request.tools[0].parameter_definitions["query"].is_required is True


def test_oci_cohere_completion_formats_tool_calls(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["cohere_tool_call"](
        "search_docs", {"query": "Oracle Cloud"}
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["cohere_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["docs_prompt"])}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search documentation",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )

    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "search_docs"
    assert json.loads(result[0]["function"]["arguments"]) == {
        "query": "Oracle Cloud"
    }


def test_oci_cohere_request_excludes_trailing_tool_messages_from_chat_history(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["cohere_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    chat_history, tool_results, message_text = llm._build_cohere_chat_history(
        [
            {"role": "user", "content": "First question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "lookup_a", "arguments": '{"query":"a"}'},
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "lookup_b", "arguments": '{"query":"b"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup_a", "content": "A"},
            {"role": "tool", "tool_call_id": "call_2", "name": "lookup_b", "content": "B"},
        ]
    )

    assert len(chat_history) == 2
    assert chat_history[0].message == "First question"
    assert len(chat_history[1].tool_calls) == 2
    assert tool_results is not None
    assert len(tool_results) == 2
    assert tool_results[0].call.name == "lookup_a"
    assert tool_results[1].call.name == "lookup_b"
    assert message_text == ""


def test_oci_completion_returns_tool_calls_for_executor(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["tool_call"](
        "get_weather", '{"city":"Paris"}'
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["weather_prompt"])}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["arguments"] == '{"city":"Paris"}'


def test_oci_completion_supports_generic_tool_controls(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["llama_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        tool_choice="get_weather",
        parallel_tool_calls=True,
        tool_result_guidance=True,
    )

    request = llm._build_chat_request(
        [
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": "Weather for Paris: sunny",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    assert request.is_parallel_tool_calls is True
    assert request.tool_choice.name == "get_weather"
    assert request.messages[-1].content[0].text.startswith(
        "You have received tool results above."
    )


@pytest.mark.parametrize(
    ("tool_choice", "expected_class_name", "expected_name"),
    [
        ("auto", "ToolChoiceAuto", None),
        ("none", "ToolChoiceNone", None),
        ("required", "ToolChoiceRequired", None),
        (True, "ToolChoiceRequired", None),
        (False, "ToolChoiceNone", None),
        ({"type": "function", "function": {"name": "search_docs"}}, "ToolChoiceFunction", "search_docs"),
    ],
)
def test_oci_completion_formats_tool_choice_variants(
    patch_oci_module,
    oci_unit_values: dict[str, object],
    tool_choice: object,
    expected_class_name: str,
    expected_name: str | None,
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        tool_choice=tool_choice,
    )

    formatted = llm._build_tool_choice()

    assert formatted.__class__.__name__ == expected_class_name
    if expected_name is not None:
        assert formatted.name == expected_name


def test_oci_completion_rejects_parallel_tools_for_cohere(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    llm = OCICompletion(
        model=str(oci_unit_values["cohere_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
        parallel_tool_calls=True,
    )

    with pytest.raises(ValueError, match="do not support parallel_tool_calls"):
        llm._build_chat_request(
            [{"role": "user", "content": str(oci_unit_values["search_prompt"])}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "description": "Search documentation",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )


def test_oci_completion_executes_tool_calls_recursively(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        oci_response_factories["tool_call"]("get_weather", '{"city":"Paris"}'),
        oci_response_factories["chat"]("Sunny in Paris"),
    ]
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = llm.call(
        [{"role": "user", "content": str(oci_unit_values["weather_prompt"])}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        available_functions={
            "get_weather": lambda city: f"Weather for {city}: sunny",
        },
    )

    assert result == "Sunny in Paris"
    assert fake_client.chat.call_count == 2
    second_request = fake_client.chat.call_args_list[1].args[0]
    assert second_request.chat_request.messages[1].tool_calls[0].name == "get_weather"
    assert second_request.chat_request.messages[2].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_oci_completion_acall_delegates_to_call(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"](
        "Hello from OCI async"
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    result = await llm.acall(
        [{"role": "user", "content": str(oci_unit_values["hello_prompt"])}]
    )

    assert result == "Hello from OCI async"
    assert fake_client.chat.call_count == 1

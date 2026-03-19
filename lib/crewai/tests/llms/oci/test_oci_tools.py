"""Unit tests for OCI provider tool calling (mocked SDK)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_tool_call_response(tool_name: str = "get_weather", args: dict | None = None) -> MagicMock:
    """Build a fake OCI response with a generic tool call."""
    tc = MagicMock()
    tc.id = "tc_001"
    tc.name = tool_name
    tc.arguments = json.dumps(args or {"city": "NYC"})

    message = MagicMock()
    message.content = [MagicMock(text="")]
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    chat_response = MagicMock()
    chat_response.choices = [choice]
    chat_response.finish_reason = None
    chat_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


def _make_text_response(text: str = "The weather is sunny.") -> MagicMock:
    """Build a fake OCI text response (used after tool execution)."""
    text_part = MagicMock()
    text_part.text = text

    message = MagicMock()
    message.content = [text_part]
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    chat_response = MagicMock()
    chat_response.choices = [choice]
    chat_response.finish_reason = None
    chat_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10, total_tokens=30)

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


def _make_cohere_tool_call_response(tool_name: str = "get_weather") -> MagicMock:
    """Build a fake OCI Cohere response with a tool call."""
    tc = MagicMock()
    tc.name = tool_name
    tc.parameters = {"city": "NYC"}

    chat_response = MagicMock()
    chat_response.text = ""
    chat_response.tool_calls = [tc]
    chat_response.finish_reason = "COMPLETE"
    chat_response.usage = MagicMock(prompt_tokens=8, completion_tokens=4, total_tokens=12)

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city name"}},
                "required": ["city"],
            },
        },
    }
]


def test_oci_completion_returns_tool_calls_for_executor(
    patch_oci_module, oci_unit_values
):
    """When no available_functions, tool calls should be returned raw."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = _make_tool_call_response()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=SAMPLE_TOOLS,
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "get_weather"


def test_oci_completion_executes_tool_calls_recursively(
    patch_oci_module, oci_unit_values
):
    """With available_functions, tool should be executed and model re-called."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    # First call returns tool call, second call returns text
    fake_client.chat.side_effect = [
        _make_tool_call_response(),
        _make_text_response("It is sunny in NYC."),
    ]
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client

    def mock_get_weather(city: str) -> str:
        return f"Sunny in {city}"

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "Weather in NYC?"}],
        tools=SAMPLE_TOOLS,
        available_functions={"get_weather": mock_get_weather},
    )

    assert isinstance(result, str)
    assert "sunny" in result.lower() or "NYC" in result
    assert fake_client.chat.call_count == 2


def test_oci_completion_formats_generic_tools(
    patch_oci_module, oci_unit_values
):
    """_format_tools should produce FunctionDefinition for generic models."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    formatted = llm._format_tools(SAMPLE_TOOLS)

    assert len(formatted) == 1
    models = patch_oci_module.generative_ai_inference.models
    models.FunctionDefinition.assert_called_once()


def test_oci_completion_formats_cohere_tools(
    patch_oci_module, oci_unit_values
):
    """_format_tools should produce CohereTool for Cohere models."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    formatted = llm._format_tools(SAMPLE_TOOLS)

    assert len(formatted) == 1
    models = patch_oci_module.generative_ai_inference.models
    models.CohereTool.assert_called_once()


def test_oci_completion_cohere_extracts_tool_calls(
    patch_oci_module, oci_unit_values
):
    """Cohere tool calls should be normalized to CrewAI shape with generated IDs."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = _make_cohere_tool_call_response()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client

    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(
        messages=[{"role": "user", "content": "Weather?"}],
        tools=SAMPLE_TOOLS,
    )

    assert isinstance(result, list)
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["id"]  # Should have a generated UUID


def test_oci_completion_rejects_parallel_tools_for_cohere(
    patch_oci_module, oci_unit_values
):
    """Cohere models should raise if parallel_tool_calls is enabled."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
        parallel_tool_calls=True,
    )

    with pytest.raises(ValueError, match="parallel_tool_calls"):
        llm._build_chat_request(
            [{"role": "user", "content": "test"}],
            tools=SAMPLE_TOOLS,
        )


def test_oci_completion_respects_max_sequential_tool_calls(
    patch_oci_module, oci_unit_values
):
    """Should raise RuntimeError when tool depth exceeds max."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    # Always return tool calls to force recursion
    fake_client.chat.return_value = _make_tool_call_response()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client

    def mock_tool(city: str) -> str:
        return "result"

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
        max_sequential_tool_calls=2,
    )

    with pytest.raises(RuntimeError, match="max_sequential_tool_calls"):
        llm.call(
            messages=[{"role": "user", "content": "test"}],
            tools=SAMPLE_TOOLS,
            available_functions={"get_weather": mock_tool},
        )


def test_oci_completion_supports_function_calling(
    patch_oci_module, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm.supports_function_calling() is True


def test_oci_completion_filters_unknown_passthrough_params(
    patch_oci_module, oci_unit_values
):
    """Unknown additional_params should not crash the OCI SDK request."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = _make_text_response("ok")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    # Make GenericChatRequest only accept known keys
    patch_oci_module.generative_ai_inference.models.GenericChatRequest.attribute_map = {
        "messages": "messages",
        "api_format": "apiFormat",
    }

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
        additional_params={"bogus_param": "should_be_filtered"},
    )
    # Should not raise
    result = llm.call(messages=[{"role": "user", "content": "test"}])
    assert isinstance(result, str)

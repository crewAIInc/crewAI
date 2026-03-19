"""Unit tests for the OCI Generative AI provider (mocked SDK)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def test_oci_completion_is_used_when_oci_provider(patch_oci_module):
    """LLM(model='oci/...') should resolve to OCICompletion."""
    from crewai.llm import LLM

    fake_client = MagicMock()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )
    llm = LLM(
        model="oci/meta.llama-3.3-70b-instruct",
        compartment_id="ocid1.compartment.oc1..test",
    )
    from crewai.llms.providers.oci.completion import OCICompletion

    # LLM.__new__ returns the native provider instance directly
    assert isinstance(llm, OCICompletion)


@pytest.mark.parametrize(
    "model_id, expected_provider",
    [
        ("meta.llama-3.3-70b-instruct", "generic"),
        ("google.gemini-2.5-flash", "generic"),
        ("openai.gpt-4o", "generic"),
        ("xai.grok-3", "generic"),
        ("cohere.command-r-plus-08-2024", "cohere"),
    ],
)
def test_oci_completion_infers_provider_family(
    patch_oci_module, oci_unit_values, model_id, expected_provider
):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=model_id,
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm.oci_provider == expected_provider


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_oci_completion_initialization_parameters(patch_oci_module, oci_unit_values):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
        top_k=40,
    )
    assert llm.temperature == 0.7
    assert llm.max_tokens == 512
    assert llm.top_p == 0.9
    assert llm.top_k == 40
    assert llm.compartment_id == oci_unit_values["compartment_id"]


def test_oci_completion_uses_region_to_build_endpoint(patch_oci_module, oci_unit_values, monkeypatch):
    from crewai.llms.providers.oci.completion import OCICompletion

    monkeypatch.delenv("OCI_SERVICE_ENDPOINT", raising=False)
    monkeypatch.setenv("OCI_REGION", "us-ashburn-1")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert "us-ashburn-1" in llm.service_endpoint


# ---------------------------------------------------------------------------
# Basic call
# ---------------------------------------------------------------------------


def test_oci_completion_call_uses_chat_api(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]("test response")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(messages=[{"role": "user", "content": "Say hello"}])

    assert "test response" in result
    fake_client.chat.assert_called_once()


def test_oci_completion_cohere_call(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["cohere_chat"]("cohere reply")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = llm.call(messages=[{"role": "user", "content": "Hi"}])

    assert "cohere reply" in result
    fake_client.chat.assert_called_once()


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def test_oci_completion_treats_none_content_as_empty_text(
    patch_oci_module, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm._coerce_text(None) == ""


def test_oci_completion_call_normalizes_messages_once(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    """Ensure normalize is not called twice when _call_impl receives a list."""
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    call_count = 0
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    original_normalize = llm._normalize_messages

    def counting_normalize(msgs):
        nonlocal call_count
        call_count += 1
        return original_normalize(msgs)

    llm._normalize_messages = counting_normalize

    llm.call(messages=[{"role": "user", "content": "hi"}])
    # call() normalizes once, _call_impl should not normalize again
    assert call_count == 1


# ---------------------------------------------------------------------------
# OpenAI model quirks
# ---------------------------------------------------------------------------


def test_oci_openai_models_use_max_completion_tokens(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model="openai.gpt-4o",
        compartment_id=oci_unit_values["compartment_id"],
        max_tokens=1024,
    )
    request = llm._build_chat_request([{"role": "user", "content": "test"}])

    models = patch_oci_module.generative_ai_inference.models
    call_kwargs = models.GenericChatRequest.call_args
    assert call_kwargs is not None
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    assert kwargs.get("max_completion_tokens") == 1024
    assert "max_tokens" not in kwargs


def test_oci_openai_gpt5_omits_unsupported_temperature_and_stop(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]()
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model="openai.gpt-5",
        compartment_id=oci_unit_values["compartment_id"],
        temperature=0.5,
    )
    llm.stop = ["END"]
    llm._build_chat_request([{"role": "user", "content": "test"}])

    models = patch_oci_module.generative_ai_inference.models
    call_kwargs = models.GenericChatRequest.call_args[1]
    assert "temperature" not in call_kwargs
    assert "stop" not in call_kwargs


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oci_completion_acall_delegates_to_call(
    patch_oci_module, oci_response_factories, oci_unit_values
):
    from crewai.llms.providers.oci.completion import OCICompletion

    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["chat"]("async result")
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    result = await llm.acall(messages=[{"role": "user", "content": "async test"}])

    assert "async result" in result

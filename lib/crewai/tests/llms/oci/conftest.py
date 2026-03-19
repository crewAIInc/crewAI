"""Fixtures for OCI provider unit and integration tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake OCI SDK module (replaces `import oci` in unit tests)
# ---------------------------------------------------------------------------


def _make_fake_oci_module() -> MagicMock:
    """Build a lightweight mock of the OCI SDK surface used by OCICompletion."""
    oci = MagicMock()

    # Models namespace
    models = oci.generative_ai_inference.models

    # Serving modes
    models.OnDemandServingMode = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models.DedicatedServingMode = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    # Content types
    models.TextContent = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    # Message types
    for cls_name in (
        "UserMessage",
        "AssistantMessage",
        "SystemMessage",
        "CohereUserMessage",
        "CohereSystemMessage",
        "CohereChatBotMessage",
    ):
        setattr(models, cls_name, MagicMock(side_effect=lambda **kw: MagicMock(**kw)))

    # Request types
    models.GenericChatRequest = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models.CohereChatRequest = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    models.BaseChatRequest = MagicMock()
    models.BaseChatRequest.API_FORMAT_GENERIC = "GENERIC"
    models.BaseChatRequest.API_FORMAT_COHERE = "COHERE"

    # ChatDetails
    models.ChatDetails = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    # Auth helpers
    oci.config.from_file = MagicMock(return_value={"key_file": "/tmp/k", "security_token_file": "/tmp/t"})
    oci.signer.load_private_key_from_file = MagicMock(return_value="pk")
    oci.auth.signers.SecurityTokenSigner = MagicMock()
    oci.auth.signers.InstancePrincipalsSecurityTokenSigner = MagicMock()
    oci.auth.signers.get_resource_principals_signer = MagicMock()
    oci.retry.DEFAULT_RETRY_STRATEGY = "default_retry"

    # Client constructor
    oci.generative_ai_inference.GenerativeAiInferenceClient = MagicMock()

    return oci


def _make_fake_chat_response(text: str = "Hello from OCI") -> MagicMock:
    """Build a minimal OCI chat response for generic models."""
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

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    chat_response.usage = usage

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


def _make_fake_cohere_chat_response(text: str = "Hello from Cohere") -> MagicMock:
    """Build a minimal OCI chat response for Cohere models."""
    chat_response = MagicMock()
    chat_response.text = text
    chat_response.finish_reason = "COMPLETE"
    chat_response.tool_calls = None

    usage = MagicMock()
    usage.prompt_tokens = 8
    usage.completion_tokens = 4
    usage.total_tokens = 12
    chat_response.usage = usage

    response = MagicMock()
    response.data.chat_response = chat_response
    return response


@pytest.fixture()
def oci_fake_module() -> MagicMock:
    return _make_fake_oci_module()


@pytest.fixture()
def patch_oci_module(monkeypatch: pytest.MonkeyPatch, oci_fake_module: MagicMock) -> MagicMock:
    """Patch the OCI module import so no real SDK is needed."""
    monkeypatch.setattr(
        "crewai.llms.providers.oci.completion._get_oci_module",
        lambda: oci_fake_module,
    )
    return oci_fake_module


@pytest.fixture()
def oci_response_factories() -> dict[str, Any]:
    return {
        "chat": _make_fake_chat_response,
        "cohere_chat": _make_fake_cohere_chat_response,
    }


# ---------------------------------------------------------------------------
# Unit test defaults
# ---------------------------------------------------------------------------

@pytest.fixture()
def oci_unit_values() -> dict[str, str]:
    return {
        "compartment_id": "ocid1.compartment.oc1..test",
        "model": "meta.llama-3.3-70b-instruct",
        "cohere_model": "cohere.command-r-plus-08-2024",
    }


# ---------------------------------------------------------------------------
# Integration test fixtures (live OCI calls)
# ---------------------------------------------------------------------------

def _env_models(env_var: str, fallback_var: str, default: str) -> list[str]:
    """Read model list from env, supporting comma-separated values."""
    raw = os.getenv(env_var) or os.getenv(fallback_var) or default
    return [m.strip() for m in raw.split(",") if m.strip()]


def _skip_unless_live_config() -> dict[str, str]:
    """Return live config dict or skip the test."""
    compartment = os.getenv("OCI_COMPARTMENT_ID")
    if not compartment:
        pytest.skip("OCI_COMPARTMENT_ID not set — skipping live test")
    region = os.getenv("OCI_REGION")
    endpoint = os.getenv("OCI_SERVICE_ENDPOINT")
    if not region and not endpoint:
        pytest.skip("Set OCI_REGION or OCI_SERVICE_ENDPOINT for live tests")
    config: dict[str, str] = {"compartment_id": compartment}
    if endpoint:
        config["service_endpoint"] = endpoint
    if os.getenv("OCI_AUTH_TYPE"):
        config["auth_type"] = os.getenv("OCI_AUTH_TYPE", "API_KEY")
    if os.getenv("OCI_AUTH_PROFILE"):
        config["auth_profile"] = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    if os.getenv("OCI_AUTH_FILE_LOCATION"):
        config["auth_file_location"] = os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config")
    return config


@pytest.fixture(
    params=_env_models("OCI_TEST_MODELS", "OCI_TEST_MODEL", "meta.llama-3.3-70b-instruct"),
    ids=lambda m: m,
)
def oci_chat_model(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture()
def oci_live_config() -> dict[str, str]:
    return _skip_unless_live_config()

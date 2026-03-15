from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai.llm import LLM


def _simple_init_class(name: str):
    class _Simple:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _Simple.__name__ = name
    return _Simple


class FakeOCI:
    def __init__(self) -> None:
        self.retry = SimpleNamespace(DEFAULT_RETRY_STRATEGY="retry")
        self.config = SimpleNamespace(
            from_file=lambda file_location, profile_name: {
                "file_location": file_location,
                "profile_name": profile_name,
            }
        )
        self.signer = SimpleNamespace(
            load_private_key_from_file=lambda *_args, **_kwargs: "private-key"
        )
        self.auth = SimpleNamespace(
            signers=SimpleNamespace(
                SecurityTokenSigner=lambda token, key: (token, key),
                InstancePrincipalsSecurityTokenSigner=lambda: "instance-principal",
                get_resource_principals_signer=lambda: "resource-principal",
            )
        )
        self.generative_ai_inference = SimpleNamespace(
            GenerativeAiInferenceClient=MagicMock(),
            models=SimpleNamespace(
                BaseChatRequest=SimpleNamespace(
                    API_FORMAT_GENERIC="GENERIC",
                    API_FORMAT_COHERE="COHERE",
                ),
                GenericChatRequest=_simple_init_class("GenericChatRequest"),
                ChatDetails=_simple_init_class("ChatDetails"),
                OnDemandServingMode=_simple_init_class("OnDemandServingMode"),
                DedicatedServingMode=_simple_init_class("DedicatedServingMode"),
                UserMessage=_simple_init_class("UserMessage"),
                AssistantMessage=_simple_init_class("AssistantMessage"),
                SystemMessage=_simple_init_class("SystemMessage"),
                ToolMessage=_simple_init_class("ToolMessage"),
                TextContent=_simple_init_class("TextContent"),
                ImageContent=_simple_init_class("ImageContent"),
                ImageUrl=_simple_init_class("ImageUrl"),
                DocumentContent=_simple_init_class("DocumentContent"),
                DocumentUrl=_simple_init_class("DocumentUrl"),
                VideoContent=_simple_init_class("VideoContent"),
                VideoUrl=_simple_init_class("VideoUrl"),
                AudioContent=_simple_init_class("AudioContent"),
                AudioUrl=_simple_init_class("AudioUrl"),
                FunctionCall=_simple_init_class("FunctionCall"),
                FunctionDefinition=_simple_init_class("FunctionDefinition"),
                ToolChoiceAuto=_simple_init_class("ToolChoiceAuto"),
                ToolChoiceFunction=_simple_init_class("ToolChoiceFunction"),
                ToolChoiceNone=_simple_init_class("ToolChoiceNone"),
                ToolChoiceRequired=_simple_init_class("ToolChoiceRequired"),
                StreamOptions=_simple_init_class("StreamOptions"),
                CohereChatRequest=_simple_init_class("CohereChatRequest"),
                CohereUserMessage=_simple_init_class("CohereUserMessage"),
                CohereChatBotMessage=_simple_init_class("CohereChatBotMessage"),
                CohereSystemMessage=_simple_init_class("CohereSystemMessage"),
                CohereToolMessage=_simple_init_class("CohereToolMessage"),
                CohereTool=_simple_init_class("CohereTool"),
                CohereParameterDefinition=_simple_init_class(
                    "CohereParameterDefinition"
                ),
                CohereToolCall=_simple_init_class("CohereToolCall"),
                CohereToolResult=_simple_init_class("CohereToolResult"),
                CohereResponseJsonFormat=_simple_init_class(
                    "CohereResponseJsonFormat"
                ),
                ResponseJsonSchema=_simple_init_class("ResponseJsonSchema"),
                JsonSchemaResponseFormat=_simple_init_class("JsonSchemaResponseFormat"),
            ),
        )


def fake_chat_response(text: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=[SimpleNamespace(text=text)])
                    )
                ]
            )
        )
    )


def fake_tool_call_response(name: str, arguments: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=[],
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_123",
                                    name=name,
                                    arguments=arguments,
                                )
                            ],
                        )
                    )
                ]
            )
        )
    )


def fake_cohere_tool_call_response(name: str, parameters: dict[str, str]):
    return SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                text="",
                tool_calls=[SimpleNamespace(name=name, parameters=parameters)],
            )
        )
    )


def fake_stream_response(*payloads: dict[str, object]):
    return SimpleNamespace(
        data=SimpleNamespace(
            events=lambda: [
                SimpleNamespace(data=json.dumps(payload)) for payload in payloads
            ]
        )
    )


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _unit_model_defaults() -> dict[str, str]:
    return {
        "generic_model": _env("OCI_UNIT_GENERIC_MODEL", "meta.test-generic-model"),
        "generic_tool_model": _env("OCI_UNIT_OPENAI_MODEL", "openai.test-chat-model"),
        "generic_structured_model": _env(
            "OCI_UNIT_STRUCTURED_MODEL", "openai.test-structured-model"
        ),
        "gpt4_control_model": _env("OCI_UNIT_GPT4_MODEL", "openai.test-gpt4-model"),
        "gpt5_model": _env("OCI_UNIT_GPT5_MODEL", "openai.gpt-5"),
        "cohere_model": _env("OCI_UNIT_COHERE_MODEL", "cohere.test-chat-model"),
        "cohere_chat_model": _env(
            "OCI_UNIT_COHERE_CHAT_MODEL", "cohere.test-chat-model"
        ),
        "gemini_model": _env("OCI_UNIT_GEMINI_MODEL", "google.test-chat-model"),
        "llama_model": _env("OCI_UNIT_LLAMA_MODEL", "meta.test-llama-model"),
        "grok_model": _env("OCI_UNIT_GROK_MODEL", "xai.test-chat-model"),
    }


def _provider_family_cases() -> list[tuple[str, str]]:
    models = _unit_model_defaults()
    return [
        (models["gpt5_model"], "generic"),
        (models["gemini_model"], "generic"),
        (models["llama_model"], "generic"),
        (models["grok_model"], "generic"),
        (models["cohere_chat_model"], "cohere"),
    ]


def _unit_test_values() -> dict[str, object]:
    models = _unit_model_defaults()
    return {
        "compartment_id": _env("OCI_UNIT_COMPARTMENT_ID", "ocid1.compartment.oc1..test"),
        "service_endpoint": _env(
            "OCI_UNIT_SERVICE_ENDPOINT",
            "https://inference.generativeai.test-region-1.oci.oraclecloud.com",
        ),
        "region": _env("OCI_UNIT_REGION", "test-region-1"),
        "generic_model": models["generic_model"],
        "generic_tool_model": models["generic_tool_model"],
        "generic_structured_model": models["generic_structured_model"],
        "gpt4_control_model": models["gpt4_control_model"],
        "gpt5_model": models["gpt5_model"],
        "cohere_model": models["cohere_model"],
        "cohere_chat_model": models["cohere_chat_model"],
        "gemini_model": models["gemini_model"],
        "llama_model": models["llama_model"],
        "grok_model": models["grok_model"],
        "prefixed_model": f"oci/{models['generic_model']}",
        "chat_prompt": _env("OCI_UNIT_CHAT_PROMPT", "Tell me something about Oracle Cloud."),
        "hello_prompt": _env("OCI_UNIT_HELLO_PROMPT", "Say hello"),
        "json_prompt": _env("OCI_UNIT_JSON_PROMPT", "Summarize OCI in JSON."),
        "search_prompt": _env("OCI_UNIT_SEARCH_PROMPT", "Search Oracle Cloud docs"),
        "docs_prompt": _env("OCI_UNIT_DOCS_PROMPT", "Find docs about Oracle Cloud"),
        "weather_prompt": _env("OCI_UNIT_WEATHER_PROMPT", "What is the weather in Paris?"),
        "multimodal_prompt": _env("OCI_UNIT_MULTIMODAL_PROMPT", "Summarize these files"),
        "provider_family_cases": _provider_family_cases(),
    }


def _prompt_defaults() -> dict[str, str]:
    return {
        "basic": _env(
            "OCI_TEST_BASIC_PROMPT", "Reply with exactly two words about Oracle Cloud."
        ),
        "stream": _env(
            "OCI_TEST_STREAM_PROMPT", "Reply with exactly three words about Oracle Cloud."
        ),
        "async": _env(
            "OCI_TEST_ASYNC_PROMPT", "Reply with exactly two words about Oracle Cloud."
        ),
        "structured": _env(
            "OCI_TEST_STRUCTURED_PROMPT",
            "Return a short JSON summary about Oracle Cloud.",
        ),
        "tool": _env(
            "OCI_TEST_TOOL_PROMPT",
            "Use the add_numbers tool to calculate 15 + 27. Return only the final result.",
        ),
        "tool_structured": _env(
            "OCI_TEST_TOOL_STRUCTURED_PROMPT",
            "Calculate 15 + 27 using your add_numbers tool. Report the result.",
        ),
    }

OCI_ALLOWED_HOSTS = [r".*\.oci\.oraclecloud\.com"]


def _resolve_model_matrix(single_var: str, list_var: str) -> list[str | None]:
    single_model = os.getenv(single_var)
    if single_model:
        return [single_model]

    models_env = os.getenv(list_var)
    if models_env:
        return [model.strip() for model in models_env.split(",") if model.strip()]

    return [None]


def _has_oci_sdk() -> bool:
    try:
        import oci  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "oci_provider_family_case" in metafunc.fixturenames:
        provider_family_cases = _provider_family_cases()
        metafunc.parametrize(
            "oci_provider_family_case",
            provider_family_cases,
            ids=[case[0] for case in provider_family_cases],
        )

    if "oci_chat_model" in metafunc.fixturenames:
        models = _resolve_model_matrix("OCI_TEST_MODEL", "OCI_TEST_MODELS")
        metafunc.parametrize(
            "oci_chat_model",
            models,
            indirect=True,
            ids=[model or "unconfigured-chat-model" for model in models],
        )

    if "oci_tool_model" in metafunc.fixturenames:
        models = _resolve_model_matrix("OCI_TEST_TOOL_MODEL", "OCI_TEST_TOOL_MODELS")
        metafunc.parametrize(
            "oci_tool_model",
            models,
            indirect=True,
            ids=[model or "unconfigured-tool-model" for model in models],
        )

    if "oci_multimodal_model" in metafunc.fixturenames:
        models = _resolve_model_matrix(
            "OCI_TEST_MULTIMODAL_MODEL", "OCI_TEST_MULTIMODAL_MODELS"
        )
        metafunc.parametrize(
            "oci_multimodal_model",
            models,
            indirect=True,
            ids=[model or "unconfigured-multimodal-model" for model in models],
        )


@pytest.fixture
def vcr_config() -> dict[str, list[str]]:
    return {"allowed_hosts": OCI_ALLOWED_HOSTS}


@pytest.fixture
def allowed_hosts() -> list[str]:
    return [r".*"]


@pytest.fixture
def oci_unit_values() -> dict[str, object]:
    return dict(_unit_test_values())


@pytest.fixture
def oci_prompts() -> dict[str, str]:
    return dict(_prompt_defaults())


@pytest.fixture
def oci_fake_module() -> FakeOCI:
    return FakeOCI()


@pytest.fixture
def patch_oci_module(monkeypatch: pytest.MonkeyPatch, oci_fake_module: FakeOCI) -> FakeOCI:
    monkeypatch.setattr(
        "crewai.llms.providers.oci.completion._get_oci_module",
        lambda: oci_fake_module,
    )
    return oci_fake_module


@pytest.fixture
def oci_response_factories():
    return {
        "chat": fake_chat_response,
        "tool_call": fake_tool_call_response,
        "cohere_tool_call": fake_cohere_tool_call_response,
        "stream": fake_stream_response,
    }


@pytest.fixture
def oci_live_config() -> dict[str, str | None]:
    if not _has_oci_sdk():
        pytest.skip("Requires OCI SDK")

    compartment_id = os.getenv("OCI_COMPARTMENT_ID")
    region = os.getenv("OCI_TEST_REGION") or os.getenv("OCI_REGION")
    service_endpoint = os.getenv("OCI_TEST_SERVICE_ENDPOINT") or os.getenv(
        "OCI_SERVICE_ENDPOINT"
    )
    if not compartment_id or not (region or service_endpoint):
        pytest.skip(
            "Requires OCI_COMPARTMENT_ID plus OCI_REGION/OCI_TEST_REGION or OCI_SERVICE_ENDPOINT/OCI_TEST_SERVICE_ENDPOINT"
        )

    return {
        "compartment_id": compartment_id,
        "service_endpoint": service_endpoint,
        "auth_type": os.getenv("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
        "auth_file_location": os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
    }


@pytest.fixture
def oci_live_llm_factory(oci_live_config: dict[str, str | None]):
    def _factory(model: str, **kwargs: object) -> LLM:
        return LLM(
            model=f"oci/{model}",
            compartment_id=oci_live_config["compartment_id"],
            service_endpoint=oci_live_config["service_endpoint"],
            auth_type=oci_live_config["auth_type"],
            auth_profile=oci_live_config["auth_profile"],
            auth_file_location=oci_live_config["auth_file_location"],
            **kwargs,
        )

    return _factory


@pytest.fixture
def oci_chat_model(request: pytest.FixtureRequest) -> str:
    model = request.param
    if not model:
        pytest.skip("Configure OCI_TEST_MODEL or OCI_TEST_MODELS for live chat tests")
    return model


@pytest.fixture
def oci_tool_model(request: pytest.FixtureRequest) -> str:
    model = request.param
    if not model:
        pytest.skip(
            "Configure OCI_TEST_TOOL_MODEL or OCI_TEST_TOOL_MODELS for live tool tests"
        )
    return model


@pytest.fixture
def oci_multimodal_model(request: pytest.FixtureRequest) -> str:
    model = request.param
    if not model:
        pytest.skip(
            "Configure OCI_TEST_MULTIMODAL_MODEL or OCI_TEST_MULTIMODAL_MODELS for live multimodal tests"
        )
    return model


@pytest.fixture
def oci_temperature_for_model():
    def _temperature(model: str) -> float | None:
        if model.startswith("openai.gpt-5"):
            return None
        return 0

    return _temperature


@pytest.fixture
def oci_token_budget():
    def _budget(model: str, scenario: str) -> int:
        if scenario == "structured" and model.startswith("openai.gpt-5"):
            return 2048
        if scenario == "agent" and model.startswith("openai.gpt-5"):
            return 1536
        if scenario == "stream" and model.startswith("openai.gpt-5"):
            return 1536
        if scenario in {"basic", "async"} and model.startswith("openai.gpt-5"):
            return 1024
        if scenario == "agent" and model.startswith("google.gemini"):
            return 384
        if scenario in {"basic", "async", "structured"} and model.startswith(
            "google.gemini"
        ):
            return 256
        if scenario == "stream":
            return 64
        if scenario == "agent":
            return 256
        return 128

    return _budget

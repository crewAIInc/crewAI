"""A missing or placeholder LLM API key fails with ``MissingAPIKeyError``.

It used to be a bare ``ValueError`` - or, for the ``YOUR_API_KEY`` placeholder that
``crewai create flow`` writes to ``.env``, an authentication error from the provider
- which neither users nor telemetry (it records only the exception class name) could
tell apart from any other failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import TaskFailedEvent
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM, MissingAPIKeyError
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.gemini.completion import GeminiCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)
from crewai_cli.deploy.validate import DeployValidator
from pydantic import ValidationError
import pytest

from ..utils import wait_for_event_handlers


PLACEHOLDER = "YOUR_API_KEY"


@dataclass(frozen=True)
class Provider:
    env_var: str
    model: str
    llm_model: str
    native: type[BaseLLM]
    base_env: tuple[tuple[str, str], ...] = ()
    """Non-key settings the provider needs before it gets as far as the key."""
    hide_keyless_auth: bool = False
    """Hide ``azure.identity`` so Azure has no keyless fallback and needs the key."""


LAZY_PROVIDERS = [
    Provider("OPENAI_API_KEY", "gpt-4o-mini", "openai/gpt-4o-mini", OpenAICompletion),
    Provider(
        "ANTHROPIC_API_KEY",
        "claude-3-5-haiku-latest",
        "anthropic/claude-3-5-haiku-latest",
        AnthropicCompletion,
    ),
    Provider(
        "GEMINI_API_KEY", "gemini-2.0-flash", "gemini/gemini-2.0-flash", GeminiCompletion
    ),
    Provider(
        "AZURE_API_KEY",
        "gpt-4o",
        "azure/gpt-4o",
        AzureCompletion,
        base_env=(("AZURE_ENDPOINT", "https://test.openai.azure.com"),),
        hide_keyless_auth=True,
    ),
]


@contextmanager
def provider_env(provider: Provider, **env: str) -> Iterator[None]:
    """Only ``env`` and the provider's non-key settings; nothing from the real env."""
    hidden = {"azure.identity": None} if provider.hide_keyless_auth else {}
    with (
        patch.dict(os.environ, {**dict(provider.base_env), **env}, clear=True),
        patch.dict("sys.modules", hidden),
    ):
        yield


def build(provider: Provider, via_llm: bool, **kwargs: Any) -> BaseLLM:
    if via_llm:
        return LLM(model=provider.llm_model, **kwargs)
    return provider.native(model=provider.model, **kwargs)


def expected_message(provider: Provider) -> str:
    return (
        f"{provider.env_var} is required for model '{provider.model}' but is not set "
        f"or is still the placeholder 'YOUR_API_KEY'. Add "
        f"{provider.env_var}=<your-api-key> to the .env file in your project root, "
        f"then run `crewai run` again."
    )


def invoke(llm: BaseLLM, mode: str) -> Any:
    if mode == "sync":
        return llm.call("Say hi")
    return asyncio.run(llm.acall("Say hi"))


ids = [p.env_var for p in LAZY_PROVIDERS]


@pytest.mark.parametrize("provider", LAZY_PROVIDERS, ids=ids)
@pytest.mark.parametrize("via_llm", [False, True], ids=["native", "LLM"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("stream", [False, True], ids=["no-stream", "stream"])
def test_missing_key_raises_on_first_call(
    provider: Provider, via_llm: bool, mode: str, stream: bool
) -> None:
    with provider_env(provider):
        llm = build(provider, via_llm, stream=stream)
        with pytest.raises(MissingAPIKeyError) as exc_info:
            invoke(llm, mode)

    assert str(exc_info.value).startswith(expected_message(provider))
    assert exc_info.value.env_var == provider.env_var
    assert exc_info.value.model == provider.model


@pytest.mark.parametrize("provider", LAZY_PROVIDERS, ids=ids)
@pytest.mark.parametrize("via_llm", [False, True], ids=["native", "LLM"])
def test_placeholder_key_in_env_counts_as_missing(
    provider: Provider, via_llm: bool
) -> None:
    with provider_env(provider, **{provider.env_var: PLACEHOLDER}):
        llm = build(provider, via_llm)
        with pytest.raises(
            MissingAPIKeyError, match=f"{provider.env_var} is required for model"
        ):
            llm.call("Say hi")


@pytest.mark.parametrize("provider", LAZY_PROVIDERS, ids=ids)
def test_placeholder_key_passed_explicitly_counts_as_missing(
    provider: Provider,
) -> None:
    with provider_env(provider):
        llm = build(provider, via_llm=False, api_key=PLACEHOLDER)
        with pytest.raises(
            MissingAPIKeyError, match=f"{provider.env_var} is required for model"
        ):
            llm.call("Say hi")


@pytest.mark.parametrize("provider", LAZY_PROVIDERS, ids=ids)
@pytest.mark.parametrize("via_llm", [False, True], ids=["native", "LLM"])
def test_valid_key_from_env_is_used_unchanged(
    provider: Provider, via_llm: bool
) -> None:
    with provider_env(provider, **{provider.env_var: "sk-real-key"}):
        llm = build(provider, via_llm)
        assert llm._get_sync_client() is not None
        assert llm.api_key == "sk-real-key"


@pytest.mark.parametrize("provider", LAZY_PROVIDERS, ids=ids)
def test_key_set_after_construction_is_picked_up(provider: Provider) -> None:
    """An LLM built at import time with the placeholder works once the key is set."""
    with provider_env(provider, **{provider.env_var: PLACEHOLDER}):
        llm = build(provider, via_llm=True)
    with provider_env(provider, **{provider.env_var: "sk-real-key"}):
        assert llm._get_sync_client() is not None
        assert llm.api_key == "sk-real-key"


def test_missing_key_is_still_a_value_error() -> None:
    """Existing ``except ValueError`` handlers keep catching it."""
    provider = LAZY_PROVIDERS[0]
    with provider_env(provider):
        llm = build(provider, via_llm=True)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            llm.call("Say hi")


def test_azure_message_names_the_keyless_alternative() -> None:
    provider = LAZY_PROVIDERS[3]
    with provider_env(provider):
        with pytest.raises(
            MissingAPIKeyError, match=r"install azure-identity: uv add "
        ):
            build(provider, via_llm=False).call("Say hi")


def test_gemini_message_names_the_vertex_alternative() -> None:
    provider = LAZY_PROVIDERS[2]
    with provider_env(provider):
        with pytest.raises(
            MissingAPIKeyError, match="gcloud auth application-default login"
        ):
            build(provider, via_llm=False).call("Say hi")


def test_gemini_placeholder_still_allows_vertex_adc() -> None:
    """The placeholder is no key, so a configured project goes to Vertex ADC."""
    provider = LAZY_PROVIDERS[2]
    with (
        provider_env(provider, GEMINI_API_KEY=PLACEHOLDER),
        patch("google.genai.Client") as client,
    ):
        GeminiCompletion(model="gemini-2.0-flash", project="my-project")

    assert client.call_args.kwargs["vertexai"] is True
    assert "api_key" not in client.call_args.kwargs


@pytest.mark.parametrize("via_llm", [False, True], ids=["native", "LLM"])
@pytest.mark.parametrize(
    ("explicit", "env"),
    [
        (None, {"GOOGLE_API_KEY": PLACEHOLDER, "GEMINI_API_KEY": "sk-real-key"}),
        (PLACEHOLDER, {"GOOGLE_API_KEY": "sk-real-key"}),
        (PLACEHOLDER, {"GOOGLE_API_KEY": PLACEHOLDER, "GEMINI_API_KEY": "sk-real-key"}),
    ],
    ids=["google-placeholder", "explicit-placeholder", "both-placeholders"],
)
def test_gemini_placeholder_does_not_hide_a_real_key(
    explicit: str | None, env: dict[str, str], via_llm: bool
) -> None:
    provider = LAZY_PROVIDERS[2]
    kwargs = {"api_key": explicit} if explicit else {}
    with provider_env(provider, **env):
        llm = build(provider, via_llm, **kwargs)
        assert llm._get_sync_client() is not None

    assert llm.api_key == "sk-real-key"


def test_gemini_reports_other_client_errors_as_themselves() -> None:
    """Only the SDK's missing-key failure becomes ``MissingAPIKeyError``."""
    provider = LAZY_PROVIDERS[2]
    with provider_env(provider):
        llm = GeminiCompletion(
            model="gemini-2.0-flash", client_params={"http_options": {"bogus": 1}}
        )
        with pytest.raises(ValueError, match="(?i)http_?options") as exc_info:
            llm.call("Say hi")

    assert not isinstance(exc_info.value, MissingAPIKeyError)


class TestAzureResponses:
    """``api="responses"`` delegates to an OpenAI client, which has its own key lookup."""

    ENDPOINT = "https://test.openai.azure.com"

    def build(self, via_llm: bool, **kwargs: Any) -> AzureCompletion:
        if via_llm:
            llm = LLM(model="azure/gpt-4o", api="responses", **kwargs)
            assert isinstance(llm, AzureCompletion)
            return llm
        return AzureCompletion(model="gpt-4o", api="responses", **kwargs)

    @pytest.mark.parametrize("via_llm", [False, True], ids=["native", "LLM"])
    @pytest.mark.parametrize("mode", ["sync", "async"])
    @pytest.mark.parametrize("stream", [False, True], ids=["no-stream", "stream"])
    @pytest.mark.parametrize("env", [{}, {"AZURE_API_KEY": PLACEHOLDER}])
    def test_missing_or_placeholder_key_names_azure(
        self, env: dict[str, str], via_llm: bool, mode: str, stream: bool
    ) -> None:
        with patch.dict(os.environ, {"AZURE_ENDPOINT": self.ENDPOINT, **env}, clear=True):
            llm = self.build(via_llm, stream=stream)
            with pytest.raises(MissingAPIKeyError) as exc_info:
                invoke(llm, mode)

        assert exc_info.value.env_var == "AZURE_API_KEY"
        assert exc_info.value.model == "gpt-4o"

    def test_message_does_not_offer_keyless_auth(self) -> None:
        """azure-identity cannot authenticate the OpenAI delegate, so do not suggest it."""
        with patch.dict(os.environ, {"AZURE_ENDPOINT": self.ENDPOINT}, clear=True):
            with pytest.raises(MissingAPIKeyError) as exc_info:
                self.build(via_llm=False).call("Say hi")

        assert "azure-identity" not in str(exc_info.value)

    @pytest.mark.parametrize(
        ("explicit", "env"),
        [
            (None, {"AZURE_API_KEY": "sk-azure-key"}),
            (PLACEHOLDER, {"AZURE_API_KEY": "sk-azure-key"}),
            ("sk-azure-key", {"OPENAI_API_KEY": "sk-openai-key"}),
        ],
        ids=["env", "explicit-placeholder", "explicit-over-openai"],
    )
    def test_delegate_gets_the_azure_key(
        self, explicit: str | None, env: dict[str, str]
    ) -> None:
        kwargs = {"api_key": explicit} if explicit else {}
        with patch.dict(os.environ, {"AZURE_ENDPOINT": self.ENDPOINT, **env}, clear=True):
            llm = self.build(via_llm=True, **kwargs)
            delegate = llm._get_responses_delegate()
            assert delegate._get_sync_client() is not None

        assert delegate.api_key == "sk-azure-key"

    def test_openai_key_fallback_is_unchanged(self) -> None:
        """Without an Azure key the delegate still reads OPENAI_API_KEY, as before."""
        env = {
            "AZURE_ENDPOINT": self.ENDPOINT,
            "AZURE_API_KEY": PLACEHOLDER,
            "OPENAI_API_KEY": "sk-openai-key",
        }
        with patch.dict(os.environ, env, clear=True):
            delegate = self.build(via_llm=True)._get_responses_delegate()
            assert delegate._get_sync_client() is not None

        assert delegate.api_key == "sk-openai-key"

    def test_key_set_after_construction_is_picked_up(self) -> None:
        with patch.dict(os.environ, {"AZURE_ENDPOINT": self.ENDPOINT}, clear=True):
            llm = self.build(via_llm=True)
        env = {"AZURE_ENDPOINT": self.ENDPOINT, "AZURE_API_KEY": "sk-azure-key"}
        with patch.dict(os.environ, env, clear=True):
            delegate = llm._get_responses_delegate()

        assert delegate.api_key == "sk-azure-key"


def test_task_failure_reports_missing_key_as_its_type() -> None:
    """Telemetry records the task's ``error_type``, so the class name is the signal."""
    failed: list[TaskFailedEvent] = []
    provider = LAZY_PROVIDERS[0]

    with crewai_event_bus.scoped_handlers(), provider_env(provider):
        agent = Agent(
            role="greeter",
            goal="Say hi",
            backstory="You greet people.",
            llm=LLM(model="openai/gpt-4o-mini"),
            max_retry_limit=0,
        )
        task = Task(description="Say hi", expected_output="hi", agent=agent)

        @crewai_event_bus.on(TaskFailedEvent)
        def record(_source: Any, event: TaskFailedEvent) -> None:
            failed.append(event)

        with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is required"):
            Crew(agents=[agent], tasks=[task]).kickoff()
        wait_for_event_handlers()

    assert [event.error_type for event in failed] == [MissingAPIKeyError]


class TestOpenAICompatible:
    """These providers check the key while the model is validated, not on first call.

    Pydantic wraps any ``ValueError`` from a validator in a ``ValidationError``, and
    ``LLM(...)`` wraps construction errors in an ``ImportError``, so the
    ``MissingAPIKeyError`` arrives inside those, with its message intact.
    """

    MESSAGE = (
        "DEEPSEEK_API_KEY is required for model 'deepseek-chat' but is not set or is "
        "still the placeholder 'YOUR_API_KEY'."
    )

    @pytest.mark.parametrize("env", [{}, {"DEEPSEEK_API_KEY": PLACEHOLDER}])
    def test_direct_construction(self, env: dict[str, str]) -> None:
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError, match=self.MESSAGE) as exc_info:
                OpenAICompatibleCompletion(model="deepseek-chat", provider="deepseek")

        cause = exc_info.value.errors()[0]["ctx"]["error"]
        assert isinstance(cause, MissingAPIKeyError)
        assert cause.env_var == "DEEPSEEK_API_KEY"

    @pytest.mark.parametrize("env", [{}, {"DEEPSEEK_API_KEY": PLACEHOLDER}])
    def test_llm_construction(self, env: dict[str, str]) -> None:
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ImportError, match=self.MESSAGE):
                LLM(model="deepseek/deepseek-chat")

    def test_explicit_placeholder(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError, match=self.MESSAGE):
                OpenAICompatibleCompletion(
                    model="deepseek-chat", provider="deepseek", api_key=PLACEHOLDER
                )

    def test_valid_key_is_used_unchanged(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-real-key"}, clear=True):
            llm = LLM(model="deepseek/deepseek-chat")

        assert llm.api_key == "sk-real-key"

    def test_keyless_provider_ignores_placeholder(self) -> None:
        """Ollama needs no key, so the placeholder falls back to its default."""
        with patch.dict(os.environ, {"OLLAMA_API_KEY": PLACEHOLDER}, clear=True):
            llm = LLM(model="ollama/llama3")

        assert llm.api_key == "ollama"

    def test_deploy_validator_reads_the_missing_variable(self, tmp_path: Path) -> None:
        """``crewai deploy`` turns this import-time failure into a targeted warning."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ImportError) as exc_info:
                LLM(model="deepseek/deepseek-chat")

        validator = DeployValidator(project_root=tmp_path)
        validator._classify_import_error("ImportError", str(exc_info.value), tb="")

        assert [r.code for r in validator.results] == ["llm_init_missing_key"]
        assert "DEEPSEEK_API_KEY" in validator.results[0].title

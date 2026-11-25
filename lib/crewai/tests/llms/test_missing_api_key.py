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
import traceback
from typing import Any
from unittest.mock import patch

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.task_events import TaskFailedEvent
from crewai.lite_agent import LiteAgent
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM, MissingAPIKeyError
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.azure.completion import AzureCompletion
from crewai.llms.providers.gemini.completion import GeminiCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleCompletion,
)
from crewai_cli.deploy.validate import DeployValidator
from pydantic import BaseModel, ValidationError, model_validator
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
    compatible: str | None = None
    """The OpenAI-compatible provider the native class is built for."""


# Every OpenAI-compatible provider that needs a key, with a model id it routes natively.
COMPATIBLE_MODELS = {
    "openrouter": "openrouter/anthropic/claude-3-opus",
    "deepseek": "deepseek/deepseek-chat",
    "cerebras": "cerebras/llama3.1-8b",
    "dashscope": "dashscope/qwen-turbo",
    "deepinfra": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "wiro": "wiro/openai/gpt-5-4-nano",
}

COMPATIBLE_PROVIDERS = [
    Provider(
        OPENAI_COMPATIBLE_PROVIDERS[name].api_key_env,
        model.split("/", 1)[1],
        model,
        OpenAICompatibleCompletion,
        compatible=name,
    )
    for name, model in COMPATIBLE_MODELS.items()
]

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
    *COMPATIBLE_PROVIDERS,
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
    if provider.compatible:
        kwargs["provider"] = provider.compatible
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


def test_every_keyed_compatible_provider_is_covered() -> None:
    required = {
        name
        for name, config in OPENAI_COMPATIBLE_PROVIDERS.items()
        if config.api_key_required
    }
    assert required == set(COMPATIBLE_MODELS)


class TestOpenAICompatible:
    """These providers check the key on the first call, like the native ones.

    They used to check it while the model was validated, so ``LLM(...)`` raised at
    construction and an ``Agent`` built from a model string raised its own
    ``ValidationError`` around it. Decided otherwise: the model builds without a
    key, wherever it is built, and the first call raises ``MissingAPIKeyError``.
    """

    MESSAGE = (
        "DEEPSEEK_API_KEY is required for model 'deepseek-chat' but is not set or is "
        "still the placeholder 'YOUR_API_KEY'."
    )

    @pytest.mark.parametrize("provider", COMPATIBLE_PROVIDERS, ids=lambda p: p.compatible)
    @pytest.mark.parametrize("key", [None, PLACEHOLDER], ids=["unset", "placeholder"])
    def test_construction_needs_no_key(self, provider: Provider, key: str | None) -> None:
        env = {} if key is None else {provider.env_var: key}
        with provider_env(provider, **env):
            llm = build(provider, via_llm=True)
            native = build(provider, via_llm=False)

        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.api_key is None and native.api_key is None

    def test_openai_key_is_never_used_for_a_compatible_provider(self) -> None:
        provider = COMPATIBLE_PROVIDERS[1]
        with provider_env(provider, OPENAI_API_KEY="sk-openai"):
            llm = build(provider, via_llm=True)
            assert llm.api_key is None
            with pytest.raises(MissingAPIKeyError, match=self.MESSAGE):
                llm.call("Say hi")

        assert llm.api_key is None
        assert llm._client is None and llm._async_client is None

    def test_explicit_key_is_used_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            llm = LLM(model="deepseek/deepseek-chat", api_key="sk-explicit")

            assert llm._get_sync_client().api_key == "sk-explicit"

    def test_keyless_provider_ignores_placeholder(self) -> None:
        """Ollama needs no key, so the placeholder falls back to its default."""
        with patch.dict(os.environ, {"OLLAMA_API_KEY": PLACEHOLDER}, clear=True):
            llm = LLM(model="ollama/llama3")

        assert llm.api_key == "ollama"

    def test_other_construction_errors_are_still_import_errors(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ImportError, match="Error importing native provider: (?s:.*)temperature"
            ) as exc_info:
                LLM(model="deepseek/deepseek-chat", api_key="sk-key", temperature="hot")

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_agent_built_from_a_model_string(self) -> None:
        """Pinned as a decision: it builds, and its first call raises the key error.

        It used to raise ``ValidationError`` here, wrapping ``MissingAPIKeyError``.
        """
        with patch.dict(os.environ, {}, clear=True):
            agent = Agent(role="r", goal="g", backstory="b", llm="deepseek/deepseek-chat")
            with pytest.raises(MissingAPIKeyError, match=self.MESSAGE):
                agent.kickoff("Say hi")

    @pytest.mark.parametrize(
        "first_use",
        [
            pytest.param(
                lambda: Agent.model_validate(
                    {"role": "r", "goal": "g", "backstory": "b", "llm": "deepseek/deepseek-chat"}
                ).llm.call("Say hi"),
                id="Agent.model_validate",
            ),
            pytest.param(
                lambda: Agent(
                    role="r",
                    goal="g",
                    backstory="b",
                    llm=LLM(model="openai/gpt-4o-mini", api_key="sk-openai"),
                    function_calling_llm="deepseek/deepseek-chat",
                ).function_calling_llm.call("Say hi"),
                id="Agent(function_calling_llm)",
            ),
            pytest.param(
                lambda: LiteAgent(
                    role="r", goal="g", backstory="b", llm="deepseek/deepseek-chat"
                ).kickoff("Say hi"),
                id="LiteAgent",
            ),
            pytest.param(
                lambda: _crew_with_function_calling_llm().function_calling_llm.call(
                    "Say hi"
                ),
                id="Crew(function_calling_llm)",
            ),
        ],
    )
    def test_every_model_holder_builds_and_raises_on_first_use(
        self, first_use: Any
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(MissingAPIKeyError, match=self.MESSAGE):
                first_use()


def _crew_with_function_calling_llm() -> Crew:
    agent = Agent(
        role="r",
        goal="g",
        backstory="b",
        llm=LLM(model="openai/gpt-4o-mini", api_key="sk-openai"),
    )
    task = Task(description="d", expected_output="o", agent=agent)
    return Crew(agents=[agent], tasks=[task], function_calling_llm="deepseek/deepseek-chat")


class TestDeployValidatorReadsOlderCrewai:
    """``crewai deploy`` imports the project with the project's own crewai.

    crewai 1.15.34 to 1.15.37 still check these providers' keys at construction:
    ``LLM(...)`` raises ``MissingAPIKeyError`` and an ``Agent`` built from a model
    string raises a ``ValidationError`` around it. Both are still warnings that
    name the variable.
    """

    ERROR = MissingAPIKeyError("DEEPSEEK_API_KEY", "deepseek-chat")

    @staticmethod
    def classify(error: BaseException, tmp_path: Path) -> DeployValidator:
        validator = DeployValidator(project_root=tmp_path)
        validator._classify_import_error(
            type(error).__name__,
            str(error),
            tb="".join(traceback.format_exception(error)),
        )
        return validator

    def test_a_missing_key_raised_as_itself(self, tmp_path: Path) -> None:
        validator = self.classify(self.ERROR, tmp_path)

        assert [r.code for r in validator.results] == ["llm_init_missing_key"]
        assert "DEEPSEEK_API_KEY" in validator.results[0].title

    def test_a_missing_key_inside_an_agents_validation_error(
        self, tmp_path: Path
    ) -> None:
        class OlderAgent(BaseModel):
            llm: str

            @model_validator(mode="after")
            def build_llm(self) -> OlderAgent:
                raise TestDeployValidatorReadsOlderCrewai.ERROR

        with pytest.raises(ValidationError) as exc_info:
            OlderAgent(llm="deepseek/deepseek-chat")

        validator = self.classify(exc_info.value, tmp_path)

        assert [r.code for r in validator.results] == ["llm_init_missing_key"]
        assert "DEEPSEEK_API_KEY" in validator.results[0].title

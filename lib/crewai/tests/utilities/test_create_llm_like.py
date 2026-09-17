"""`create_llm_like`: what an `llm_overlay` swap carries from the declared llm.

Zero-cost: instances are built, nothing is called. The provider classes that
need an SDK this environment may not have (Azure, Gemini) are stood in for by
minimal `BaseLLM` subclasses that reproduce the one behaviour under test.
Provider classes are compared by name: the provider test files delete a module
from sys.modules and re-import it, so a class imported here can be stale.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.llm_utils import (
    GENERATION_SETTINGS,
    PROVIDER_SETTINGS,
    _configured_settings,
    create_llm_like,
)


class _BakesTheDeploymentIntoTheEndpoint(BaseLLM):
    """Azure's shape: the endpoint carries the declared model's deployment."""

    endpoint: str | None = None

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _OmitsTheCapFromItsConfig(BaseLLM):
    """Gemini's shape: `to_config_dict` never emits `max_tokens`, yet a caller's
    value is honoured — the default is `None`, so nothing was derived."""

    def to_config_dict(self) -> dict[str, Any]:
        config = super().to_config_dict()
        config.pop("max_tokens", None)
        return config

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _UserDefined(BaseLLM):
    """A subclass of the user's own, whose `provider` defaulted to openai."""

    def call(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def test_an_azure_endpoint_is_carried_as_its_resource_root() -> None:
    base = _BakesTheDeploymentIntoTheEndpoint(
        model="gpt-4o",
        provider="azure",
        endpoint="https://r.openai.azure.com/openai/deployments/gpt-4o",
    )
    carried = _configured_settings(base, PROVIDER_SETTINGS)
    assert carried["endpoint"] == "https://r.openai.azure.com"


def test_a_cap_the_caller_set_is_carried_even_when_the_class_does_not_emit_it() -> None:
    base = _OmitsTheCapFromItsConfig(model="gemini-2.5-flash", max_tokens=1000)
    assert "max_tokens" not in base.to_config_dict()
    assert _configured_settings(base, GENERATION_SETTINGS)["max_tokens"] == 1000


def test_a_value_the_targets_field_type_refuses_is_dropped_with_a_warning(
    caplog: Any,
) -> None:
    """LiteLLM's `logprobs` is an int, the OpenAI class's a bool: the swap must
    not raise inside a kickoff; it builds without the one setting and says so."""
    base = LLM(model="openai/not-a-known-model", logprobs=2, temperature=0.4)
    assert type(base) is LLM
    with caplog.at_level(logging.WARNING, logger="crewai.utilities.llm_utils"):
        built = create_llm_like("openai/gpt-4o", base)
    assert type(built).__name__ == "OpenAICompletion" and built.model == "gpt-4o"
    assert built.logprobs is None and built.temperature == 0.4
    assert any("logprobs" in rec.getMessage() for rec in caplog.records)


def test_extra_kwargs_are_carried_within_a_class_and_not_across() -> None:
    same = OpenAICompletion(model="gpt-4o-mini", api_key="k", extra_body={"a": 1})
    assert same.additional_params == {"extra_body": {"a": 1}}
    assert create_llm_like("openai/gpt-4o", same).additional_params == {
        "extra_body": {"a": 1}
    }

    other = LLM(model="openai/not-a-known-model", drop_params=True)
    assert other.additional_params == {"drop_params": True}
    assert create_llm_like("openai/gpt-4o", other).additional_params == {}


def test_credentials_follow_the_class_not_the_provider_string() -> None:
    aliased = LLM(model="claude-haiku-4-5", provider="claude", api_key="k")
    assert (
        type(aliased).__name__ == "AnthropicCompletion" and aliased.provider == "claude"
    )
    assert create_llm_like("anthropic/claude-sonnet-4-5", aliased).api_key == "k"

    litellm_openai = LLM(model="openai/not-a-known-model", api_key="k")
    assert type(litellm_openai) is LLM
    assert create_llm_like("openai/gpt-4o", litellm_openai).api_key == "k"

    user_defined = _UserDefined(model="gpt-x", api_key="k")
    assert user_defined.provider == "openai"
    assert create_llm_like("openai/gpt-4o", user_defined).api_key != "k"


def test_an_output_cap_keeps_its_meaning_under_the_targets_name() -> None:
    base = OpenAICompletion(model="gpt-4o-mini", api_key="k", max_completion_tokens=300)
    swapped = create_llm_like("anthropic/claude-haiku-4-5", base)
    assert type(swapped).__name__ == "AnthropicCompletion" and swapped.max_tokens == 300


def test_anthropic_gets_temperature_or_top_p_not_both() -> None:
    base = OpenAICompletion(
        model="gpt-4o-mini", api_key="k", temperature=0.7, top_p=0.9
    )
    swapped = create_llm_like("anthropic/claude-haiku-4-5", base)
    assert type(swapped).__name__ == "AnthropicCompletion"
    assert swapped.temperature == 0.7 and swapped.top_p is None
    only_top_p = OpenAICompletion(model="gpt-4o-mini", api_key="k", top_p=0.9)
    assert create_llm_like("anthropic/claude-haiku-4-5", only_top_p).top_p == 0.9

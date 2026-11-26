"""Offline token callback tests for dictionary and SDK-object usage."""

from typing import Any

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.utilities.token_counter_callback import TokenCalcHandler
from openai.types.completion_usage import CompletionUsage
import pytest


@pytest.mark.parametrize("as_dict", [True, False])
@pytest.mark.parametrize("cached_tokens", [0, 4])
def test_token_callback_counts_dictionary_and_object_usage(
    as_dict: bool, cached_tokens: int
) -> None:
    usage: dict[str, Any] = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }
    process = TokenProcess()
    callback = TokenCalcHandler(process)
    payload = usage if as_dict else CompletionUsage(**usage)

    callback.log_success_event({}, {"usage": payload}, 0, 0)
    callback.log_success_event({}, {"usage": payload}, 0, 0)

    assert process.prompt_tokens == 24
    assert process.completion_tokens == 6
    assert process.total_tokens == 30
    assert process.cached_prompt_tokens == cached_tokens * 2
    assert process.successful_requests == 2


@pytest.mark.parametrize("usage", [None, {}])
def test_token_callback_skips_missing_usage(usage: Any) -> None:
    process = TokenProcess()
    TokenCalcHandler(process).log_success_event({}, {"usage": usage}, 0, 0)

    assert process.get_summary().total_tokens == 0
    assert process.successful_requests == 0


def test_token_callback_without_token_process() -> None:
    TokenCalcHandler().log_success_event(
        {}, {"usage": {"prompt_tokens": 12, "completion_tokens": 3}}, 0, 0
    )

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from crewai.llms.context_window import (
    ANTHROPIC_CONTEXT_WINDOWS,
    BEDROCK_CONTEXT_WINDOWS,
    GEMINI_CONTEXT_WINDOWS,
    CONTEXT_WINDOW_USAGE_RATIO,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    LLM_CONTEXT_WINDOW_SIZES,
    bare_context_windows,
    context_windows_from_catalog,
    _chunk_messages,
    summarize_messages,
    resolve_context_window_size,
)
from crewai.utilities.message_utils import build_summary_prompt
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


def test_resolver_uses_the_exact_model_id() -> None:
    sizes = {"gpt-5": 1_047_576, "gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-mini", sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bare_lookup_uses_the_full_catalog_when_the_provider_map_misses() -> None:
    result = resolve_context_window_size(
        "us.anthropic.claude-sonnet-4-5-v1:0",
        {},
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_unrecognized_provider_prefix_does_not_inherit_the_bare_window() -> None:
    result = resolve_context_window_size(
        "acme/gpt-5.6-luna",
        LLM_CONTEXT_WINDOW_SIZES,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_does_not_match_a_different_model_family() -> None:
    sizes = {"gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-nano", sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(272_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_uses_default_for_an_unknown_model() -> None:
    result = resolve_context_window_size(
        "unknown-model", {}, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolver_uses_a_published_limit_as_given() -> None:
    result = resolve_context_window_size(
        "test-model", {"test-model": 500}, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(500 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bedrock_claude_uses_its_documented_context_window() -> None:
    result = resolve_context_window_size(
        "us.anthropic.claude-sonnet-4-6",
        BEDROCK_CONTEXT_WINDOWS,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(1_000_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_bedrock_regional_aliases_preserve_the_base_model_context_window() -> None:
    result = resolve_context_window_size(
        "us.meta.llama3-3-70b-instruct-v1:0",
        BEDROCK_CONTEXT_WINDOWS,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
    )

    assert result == int(128_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_litellm_map_includes_the_openai_gpt5_family() -> None:
    result = resolve_context_window_size(
        "gpt-5", LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
    )

    assert result == int(272_000 * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "raw_context_window"),
    [
        ("o1", 200_000),
        ("o1-pro", 200_000),
        ("o3", 200_000),
        ("gemini-3.1-flash-lite", 1_048_576),
        ("gemini-3.1-pro-preview", 1_048_576),
        ("gemini-3-flash-preview", 1_048_576),
    ],
)
def test_affected_model_ids_use_their_specific_context_windows(
    model: str, raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "sizes", "raw_context_window"),
    [
        ("gpt-6-astra", LLM_CONTEXT_WINDOW_SIZES, 922_000),
        ("gpt-5.6-cyber", LLM_CONTEXT_WINDOW_SIZES, 400_000),
        ("gpt-3.5-turbo", LLM_CONTEXT_WINDOW_SIZES, 16_385),
        ("chat-latest", LLM_CONTEXT_WINDOW_SIZES, 400_000),
        ("azure_ai/gpt-oss-120b", LLM_CONTEXT_WINDOW_SIZES, 131_072),
        ("gpt-5.4-nano", LLM_CONTEXT_WINDOW_SIZES, 272_000),
        ("gemini-3.7-flash", LLM_CONTEXT_WINDOW_SIZES, 1_048_576),
        ("gemma-3-27b-it", GEMINI_CONTEXT_WINDOWS, 131_072),
        ("gemma-4-31b-it", GEMINI_CONTEXT_WINDOWS, 262_144),
        ("amazon.nova-2-lite-v1:0", BEDROCK_CONTEXT_WINDOWS, 1_000_000),
        ("global.moonshotai.kimi-k3", BEDROCK_CONTEXT_WINDOWS, 1_000_000),
        # 200k: LiteLLM's catalog had 1M here until its refresh for 1.15.28
        # corrected it to the model's documented window.
        (
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            BEDROCK_CONTEXT_WINDOWS,
            200_000,
        ),
        ("us.anthropic.claude-opus-4-1-20250805-v1:0", BEDROCK_CONTEXT_WINDOWS, 200_000),
        (
            "meta.llama4-scout-17b-instruct-v1:0",
            BEDROCK_CONTEXT_WINDOWS,
            10_000_000,
        ),
    ],
)
def test_new_catalog_models_use_their_documented_context_windows(
    model: str, sizes: dict[str, int], raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, sizes, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


def test_packaged_litellm_catalog_loads_known_models() -> None:
    assert LLM_CONTEXT_WINDOW_SIZES["gpt-4"] == 8192
    assert ANTHROPIC_CONTEXT_WINDOWS["claude-sonnet-4-6"] == 1_000_000
    assert len(LLM_CONTEXT_WINDOW_SIZES) > 1000


def test_catalog_parser_keeps_published_integer_limits() -> None:
    windows = context_windows_from_catalog(
        {
            "sample_spec": {"max_input_tokens": "max input tokens"},
            "too-small": {"litellm_provider": "openai", "max_input_tokens": 77},
            "not-a-number": {
                "litellm_provider": "openai",
                "max_input_tokens": "8192",
            },
            "huge": {
                "litellm_provider": "openai",
                "max_input_tokens": 20_000_000,
            },
            "openai/gpt-example": {
                "litellm_provider": "openai",
                "max_input_tokens": 16_000,
            },
        }
    )

    assert "sample_spec" not in windows
    assert windows["too-small"] == 77
    assert "not-a-number" not in windows
    assert windows["huge"] == 20_000_000
    assert windows["openai/gpt-example"] == 16_000
    assert windows["gpt-example"] == 16_000


def test_contested_provider_aliases_are_left_to_the_bare_index() -> None:
    windows = context_windows_from_catalog(
        {
            "azure/contested-model": {
                "litellm_provider": "azure",
                "max_input_tokens": 32_000,
            },
            "azure_ai/contested-model": {
                "litellm_provider": "azure_ai",
                "max_input_tokens": 128_000,
            },
        }
    )

    assert windows["azure/contested-model"] == 32_000
    assert windows["azure_ai/contested-model"] == 128_000
    assert "contested-model" not in windows
    assert resolve_context_window_size(
        "contested-model", windows, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(DEFAULT_CONTEXT_WINDOW_SIZE * CONTEXT_WINDOW_USAGE_RATIO)


@pytest.mark.parametrize(
    ("model", "raw_context_window"),
    [
        ("us.anthropic.claude-sonnet-4-5-v1:0", 200_000),
        ("eu.anthropic.claude-sonnet-4-5-v1:0", 200_000),
        ("us.anthropic.claude-opus-4-6-v1:0", 1_000_000),
        ("us.anthropic.claude-opus-4-5-v1:0", 200_000),
        ("us.anthropic.claude-haiku-4-5-v1:0", 200_000),
        ("amazon.nova-lite-v1:0:24k", 300_000),
        ("anthropic.claude-v2:1:200k", 100_000),
    ],
)
def test_bare_model_name_matches_catalog_variants(
    model: str, raw_context_window: int
) -> None:
    assert resolve_context_window_size(
        model, BEDROCK_CONTEXT_WINDOWS, default=DEFAULT_CONTEXT_WINDOW_SIZE
    ) == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)


def test_bare_context_windows_uses_the_majority_limit() -> None:
    windows = bare_context_windows(
        {
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": 1_000_000,
        }
    )

    assert windows["claude-sonnet-4-5"] == 200_000


def test_bare_context_windows_uses_the_smaller_limit_when_votes_tie() -> None:
    windows = bare_context_windows(
        {
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 1_000_000,
        }
    )

    assert windows["claude-sonnet-4-5"] == 200_000


def test_summary_chunking_uses_the_rendered_request_token_estimator() -> None:
    """Prompt overhead affects the chunk boundary, just as it does at call time."""

    class SummaryCountingLLM:
        def __init__(self) -> None:
            self.requests = []

        def get_input_tokens_count(self, messages):
            self.requests.append(messages)
            return 100 + messages[-1]["content"].count("x")

    llm = SummaryCountingLLM()
    messages = [
        {"role": "user", "content": "x" * 20},
        {"role": "user", "content": "x" * 20},
    ]

    chunks = _chunk_messages(llm, messages, max_tokens=125)

    assert chunks == [[messages[0]], [messages[1]]]
    assert all(
        llm.get_input_tokens_count(build_summary_prompt(chunk)) <= 125
        for chunk in chunks
    )
    assert all(len(request) == 2 for request in llm.requests)


def _summary_llm(reply: str = "<summary>Summary</summary>") -> MagicMock:
    llm = MagicMock()
    llm.get_context_window_size.return_value = 1_000
    llm.get_input_tokens_count.return_value = 1
    llm.acall = AsyncMock(return_value=reply)
    return llm


def test_summary_preserves_system_messages_and_merged_files() -> None:
    first, second = MagicMock(), MagicMock()
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "First", "files": {"a.pdf": first}},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Second", "files": {"b.pdf": second}},
    ]

    summarize_messages(_summary_llm(), messages, [], verbose=False)

    assert messages[0] == {"role": "system", "content": "System"}
    assert messages[1]["files"] == {"a.pdf": first, "b.pdf": second}


def test_summary_prompt_preserves_tool_and_multimodal_text() -> None:
    llm = _summary_llm()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Question"}]},
        {"role": "tool", "name": "search", "content": "Result"},
    ]

    summarize_messages(llm, messages, [], verbose=False)

    prompt = llm.acall.call_args.args[0][1]["content"]
    assert "[USER]: Question" in prompt
    assert "[TOOL_RESULT (search)]: Result" in prompt


def test_summary_extracts_structured_summary_tag() -> None:
    messages = [{"role": "user", "content": "Question"}]

    summarize_messages(
        _summary_llm("before <summary>Extracted</summary> after"),
        messages,
        [],
        verbose=False,
    )

    assert "Extracted" in messages[0]["content"]
    assert "before" not in messages[0]["content"]


def test_summary_propagates_provider_context_errors() -> None:
    llm = _summary_llm()
    llm.acall = AsyncMock(side_effect=Exception("context length exceeded"))

    with pytest.raises(Exception, match="context length exceeded"):
        summarize_messages(
            llm, [{"role": "user", "content": "Question"}], [], verbose=False
        )


def test_summary_limits_parallel_requests_to_four() -> None:
    class ConcurrentSummaryLLM:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        def get_context_window_size(self) -> int:
            return 25

        def get_input_tokens_count(self, messages) -> int:
            return messages[-1]["content"].count("§")

        async def acall(self, messages, callbacks):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return "<summary>Summary</summary>"

    llm = ConcurrentSummaryLLM()
    summarize_messages(
        llm,
        [{"role": "user", "content": "§" * 20} for _ in range(5)],
        [],
        verbose=False,
    )

    assert llm.max_active == 4


def test_summary_chunking_splits_a_single_oversized_text_message() -> None:
    class CountingLLM:
        def get_input_tokens_count(self, messages) -> int:
            return messages[-1]["content"].count("§")

    message = {"role": "user", "content": "§" * 50}

    chunks = _chunk_messages(CountingLLM(), [message], max_tokens=20)

    assert "".join(
        part["content"] for chunk in chunks for part in chunk
    ) == message["content"]
    assert max(
        len(part["content"]) for chunk in chunks for part in chunk
    ) == 15
    assert all(
        CountingLLM().get_input_tokens_count(build_summary_prompt(chunk)) <= 20
        for chunk in chunks
    )


def test_summary_chunking_handles_non_additive_prompt_overhead() -> None:
    class NonAdditiveCountingLLM:
        def get_input_tokens_count(self, messages) -> int:
            message_count = messages[-1]["content"].count("[USER]:")
            return message_count * 10 + (5 if message_count > 1 else 0)

    messages = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]

    chunks = _chunk_messages(NonAdditiveCountingLLM(), messages, max_tokens=20)

    assert chunks == [[messages[0]], [messages[1]]]


def test_summary_chunking_rejects_unsplittable_oversized_content() -> None:
    class CountingLLM:
        def get_input_tokens_count(self, messages) -> int:
            return 100 if "[Called tools: huge]" in messages[-1]["content"] else 1

    with pytest.raises(LLMContextLengthExceededError, match="no text"):
        _chunk_messages(
            CountingLLM(),
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "huge"}}],
                }
            ],
            max_tokens=10,
        )


ONE_MILLION_TOKENS = 1_048_576
USABLE_ONE_MILLION = int(ONE_MILLION_TOKENS * CONTEXT_WINDOW_USAGE_RATIO)


def _unlisted_model() -> str:
    """A model id the catalog does not know, unique so no earlier test warned for it."""
    return f"gemini-4-argon-{uuid.uuid4().hex}"


def _fallback_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "crewai.llms.context_window"
        and record.levelno == logging.WARNING
    ]


def test_resolver_prefers_a_configured_window_over_the_catalog() -> None:
    sizes = {"gpt-5.4-mini": 200_000}

    result = resolve_context_window_size(
        "gpt-5.4-mini",
        sizes,
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
        configured=ONE_MILLION_TOKENS,
    )

    assert result == USABLE_ONE_MILLION


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"model": "openrouter/google/gemini-4-argon"}, id="openrouter"),
        pytest.param({"model": "openai/gpt-4o"}, id="openai"),
        pytest.param({"model": "anthropic/claude-haiku-4-5"}, id="anthropic"),
        pytest.param({"model": "gemini/gemini-2.5-flash"}, id="gemini"),
        pytest.param({"model": "azure/gpt-4o"}, id="azure"),
        pytest.param(
            {"model": "bedrock/anthropic.claude-sonnet-4-6"}, id="bedrock"
        ),
        pytest.param({"model": "gpt-4o", "is_litellm": True}, id="litellm"),
    ],
)
def test_explicit_context_window_size_is_honored_by_every_provider(
    kwargs: dict[str, object],
) -> None:
    from crewai import LLM

    llm = LLM(context_window_size=ONE_MILLION_TOKENS, **kwargs)

    assert llm.get_context_window_size() == USABLE_ONE_MILLION
    assert "context_window_size" not in llm.additional_params


def test_explicit_context_window_size_is_honored_when_building_a_native_class() -> (
    None
):
    from crewai.llms.providers.openai_compatible.completion import (
        OpenAICompatibleCompletion,
    )

    llm = OpenAICompatibleCompletion(
        model="google/gemini-4-argon",
        provider="openrouter",
        context_window_size=ONE_MILLION_TOKENS,
    )

    assert llm.get_context_window_size() == USABLE_ONE_MILLION


def test_explicit_context_window_size_is_not_sent_to_the_provider() -> None:
    from crewai import LLM

    llm = LLM(model="openai/gpt-4o", context_window_size=ONE_MILLION_TOKENS)

    params = llm._prepare_completion_params(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert "context_window_size" not in params


def test_custom_base_llm_honors_explicit_context_window_size() -> None:
    from crewai.llms.base_llm import BaseLLM

    class CustomLLM(BaseLLM):
        def call(self, *args: object, **kwargs: object) -> str:
            return "ok"

    assert CustomLLM(model="custom").get_context_window_size() == 4096
    assert (
        CustomLLM(
            model="custom", context_window_size=ONE_MILLION_TOKENS
        ).get_context_window_size()
        == USABLE_ONE_MILLION
    )


@pytest.mark.parametrize(
    ("model", "raw_context_window"),
    [
        ("openai/gpt-4o", 128_000),
        ("anthropic/claude-haiku-4-5", 200_000),
        ("bedrock/amazon.nova-2-lite-v1:0", 1_000_000),
        ("azure/gpt-4o", 128_000),
    ],
)
def test_catalog_models_keep_their_window_without_a_warning(
    model: str, raw_context_window: int, caplog: pytest.LogCaptureFixture
) -> None:
    from crewai import LLM

    with caplog.at_level(logging.WARNING, logger="crewai.llms.context_window"):
        size = LLM(model=model).get_context_window_size()

    assert size == int(raw_context_window * CONTEXT_WINDOW_USAGE_RATIO)
    assert _fallback_warnings(caplog) == []


def test_unlisted_openrouter_model_falls_back_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crewai import LLM

    model = _unlisted_model()

    with caplog.at_level(logging.WARNING, logger="crewai.llms.context_window"):
        first = LLM(model=f"openrouter/google/{model}")
        sizes = [first.get_context_window_size(), first.get_context_window_size()]
        LLM(model=f"openrouter/google/{model}").get_context_window_size()

    assert sizes == [6963, 6963]
    warnings = _fallback_warnings(caplog)
    assert len(warnings) == 1
    assert f"google/{model}" in warnings[0]
    assert "8,192 tokens (6,963 usable)" in warnings[0]
    assert "context_window_size=<tokens>" in warnings[0]


@pytest.mark.parametrize(
    ("prefix", "kwargs"),
    [
        pytest.param("openai/gpt-", {}, id="openai"),
        pytest.param("anthropic/claude-", {}, id="anthropic"),
        pytest.param("", {"is_litellm": True}, id="litellm"),
    ],
)
def test_unlisted_model_warns_on_every_provider_path(
    prefix: str, kwargs: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    from crewai import LLM

    model = f"{prefix}{_unlisted_model()}"

    with caplog.at_level(logging.WARNING, logger="crewai.llms.context_window"):
        LLM(model=model, **kwargs).get_context_window_size()

    warnings = _fallback_warnings(caplog)
    assert len(warnings) == 1
    assert model.removeprefix("openai/").removeprefix("anthropic/") in warnings[0]


def test_each_unlisted_model_gets_its_own_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first, second = _unlisted_model(), _unlisted_model()

    with caplog.at_level(logging.WARNING, logger="crewai.llms.context_window"):
        for model in (first, second, first):
            resolve_context_window_size(
                model, {}, default=DEFAULT_CONTEXT_WINDOW_SIZE
            )

    warnings = _fallback_warnings(caplog)
    assert len(warnings) == 2
    assert first in warnings[0]
    assert second in warnings[1]


def test_explicit_context_window_size_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from crewai import LLM

    llm = LLM(
        model=f"openrouter/google/{_unlisted_model()}",
        context_window_size=ONE_MILLION_TOKENS,
    )

    with caplog.at_level(logging.WARNING, logger="crewai.llms.context_window"):
        llm.get_context_window_size()

    assert _fallback_warnings(caplog) == []


def test_explicit_context_window_size_moves_the_deferral_and_compaction_gates() -> (
    None
):
    from crewai import LLM

    model = f"openrouter/google/{_unlisted_model()}"
    tool_result = "x" * 4 * 4_000  # about 4,000 tokens by the 4-chars estimate
    fallback = LLM(model=model)
    configured = LLM(model=model, context_window_size=ONE_MILLION_TOKENS)

    assert fallback.is_tool_result_overflowing_threshold(tool_result)[0] is True
    assert configured.is_tool_result_overflowing_threshold(tool_result)[0] is False
    assert fallback.is_context_window_overflowing_threshold(tool_result * 2) is True
    assert (
        configured.is_context_window_overflowing_threshold(tool_result * 2) is False
    )


@pytest.mark.parametrize("value", [-1, -1_048_576])
def test_negative_context_window_size_is_rejected(value: int) -> None:
    from crewai.llms.providers.openai.completion import OpenAICompletion

    with pytest.raises(
        ValueError,
        match=f"context_window_size must be a positive number of tokens, got {value}",
    ):
        OpenAICompletion(model="gpt-4o", context_window_size=value)


@pytest.mark.parametrize("value", [True, "1048576", 1.5])
def test_non_integer_context_window_size_is_rejected(value: object) -> None:
    from crewai.llms.providers.openai.completion import OpenAICompletion

    with pytest.raises(ValueError, match="Input should be a valid integer"):
        OpenAICompletion(model="gpt-4o", context_window_size=value)


def test_zero_context_window_size_from_an_older_dump_means_unset() -> None:
    from crewai import LLM

    llm = LLM(model="gpt-4o", is_litellm=True, context_window_size=0)

    assert llm.context_window_size is None
    assert llm.get_context_window_size() == int(128_000 * CONTEXT_WINDOW_USAGE_RATIO)


def test_resolved_catalog_window_is_not_written_back_into_the_field() -> None:
    from crewai import LLM

    llm = LLM(model="gpt-4o", is_litellm=True)
    llm.get_context_window_size()

    assert llm.context_window_size is None
    assert llm.model_dump()["context_window_size"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"model": "openrouter/google/gemini-4-argon"}, id="openrouter"),
        pytest.param({"model": "anthropic/claude-haiku-4-5"}, id="anthropic"),
        pytest.param({"model": "gpt-4o", "is_litellm": True}, id="litellm"),
    ],
)
def test_explicit_context_window_size_survives_an_agent_checkpoint_round_trip(
    kwargs: dict[str, object],
) -> None:
    from crewai import LLM, Agent

    agent = Agent(
        role="Researcher",
        goal="Research",
        backstory="Researches.",
        llm=LLM(context_window_size=ONE_MILLION_TOKENS, **kwargs),
    )

    restored = Agent.model_validate(
        agent.model_dump(mode="json"), context={"from_checkpoint": True}
    )

    assert restored.llm.get_context_window_size() == USABLE_ONE_MILLION


@pytest.mark.parametrize(
    ("model", "full_window"),
    [
        pytest.param("gpt-4o", 128_000, id="catalog"),
        pytest.param("anthropic/claude-haiku-4-5", 200_000, id="provider-prefixed"),
        pytest.param(
            "openrouter/google/gemini-4-argon",
            DEFAULT_CONTEXT_WINDOW_SIZE,
            id="fallback",
        ),
    ],
)
def test_checkpoint_with_a_pre_change_cached_window_restores_the_catalog_window(
    model: str, full_window: int
) -> None:
    from crewai import LLM, Agent

    usable = int(full_window * CONTEXT_WINDOW_USAGE_RATIO)
    agent = Agent(
        role="Researcher",
        goal="Research",
        backstory="Researches.",
        llm=LLM(model=model, is_litellm=True),
    )
    checkpoint = agent.model_dump(mode="json")
    # Through 1.15.32, get_context_window_size() cached its usable result here.
    checkpoint["llm"]["context_window_size"] = usable

    restored = Agent.model_validate(checkpoint, context={"from_checkpoint": True})

    assert restored.llm.get_context_window_size() == usable
    assert restored.llm.model_dump()["context_window_size"] is None


def test_explicit_context_window_size_survives_to_config_dict() -> None:
    from crewai import LLM

    llm = LLM(model="anthropic/claude-haiku-4-5", context_window_size=ONE_MILLION_TOKENS)

    rebuilt = LLM(**llm.to_config_dict())

    assert rebuilt.get_context_window_size() == USABLE_ONE_MILLION


def test_explicit_context_window_size_is_honored_from_an_llm_config_dict() -> None:
    from crewai.utilities.llm_utils import create_llm

    llm = create_llm(
        {
            "model": "openrouter/google/gemini-4-argon",
            "context_window_size": ONE_MILLION_TOKENS,
        }
    )

    assert llm is not None
    assert llm.get_context_window_size() == USABLE_ONE_MILLION

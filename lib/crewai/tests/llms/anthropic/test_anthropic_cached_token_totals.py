"""Anthropic cached input tokens must be counted in the reported totals.

Anthropic reports ``input_tokens`` NET of the prompt cache and bills
``cache_read_input_tokens`` and ``cache_creation_input_tokens`` separately, and
it never sends a total. Every other provider CrewAI supports reports a prompt
count that already contains its cached portion: OpenAI nests ``cached_tokens``
inside ``prompt_tokens``, and Gemini and Bedrock send an authoritative total.

So mapping Anthropic's raw ``input_tokens`` onto ``prompt_tokens`` makes the
same field mean two different things depending on the provider, and
``total_tokens`` comes out low on Anthropic alone. Nothing raises: the number
is simply smaller than what was billed. ``UsageMetrics.from_provider_dict``
states the invariant these tests defend, that per-LLM totals, flow-level
aggregation and OTel spans "agree on every provider".
"""

from types import SimpleNamespace

from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.types.usage_metrics import UsageMetrics


def _usage(**kwargs):
    """Build an Anthropic-shaped usage object."""
    return SimpleNamespace(**kwargs)


def _extract(usage):
    return AnthropicCompletion._extract_anthropic_token_usage(
        SimpleNamespace(usage=usage)
    )


class TestAnthropicCachedTokensInTotals:
    def test_cache_read_tokens_are_counted_in_the_total(self):
        """200 cached + 50 fresh input + 100 output is 350 billed tokens."""
        result = _extract(
            _usage(
                input_tokens=50,
                output_tokens=100,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=0,
            )
        )
        assert result["total_tokens"] == 350

    def test_cache_creation_tokens_are_counted_in_the_total(self):
        """Cache writes are billed too, at a premium, so they belong in the total."""
        result = _extract(
            _usage(
                input_tokens=50,
                output_tokens=100,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=200,
            )
        )
        assert result["total_tokens"] == 350

    def test_cached_prompt_tokens_stay_a_subset_of_the_prompt_count(self):
        """Matches OpenAI, where cached_tokens is contained by prompt_tokens.

        If the cached count were additive rather than a subset, any consumer
        adding them together would double count.
        """
        result = _extract(
            _usage(
                input_tokens=50,
                output_tokens=100,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=25,
            )
        )
        assert result["input_tokens"] == 275
        assert result["cached_prompt_tokens"] == 200
        assert result["cache_creation_tokens"] == 25
        assert result["cached_prompt_tokens"] <= result["input_tokens"]

    def test_totals_agree_across_providers_for_the_same_billed_work(self):
        """The invariant from_provider_dict's own docstring promises.

        The same 250 input tokens, 200 of them cached, and 100 output tokens,
        reported in Anthropic's shape and in OpenAI's shape, must produce the
        same UsageMetrics total.
        """
        anthropic = UsageMetrics.from_provider_dict(
            _extract(
                _usage(
                    input_tokens=50,
                    output_tokens=100,
                    cache_read_input_tokens=200,
                    cache_creation_input_tokens=0,
                )
            )
        )
        openai = UsageMetrics.from_provider_dict(
            {
                "prompt_tokens": 250,
                "completion_tokens": 100,
                "total_tokens": 350,
                "prompt_tokens_details": {"cached_tokens": 200},
            }
        )
        assert anthropic.total_tokens == openai.total_tokens
        assert anthropic.prompt_tokens == openai.prompt_tokens
        assert anthropic.cached_prompt_tokens == openai.cached_prompt_tokens

    def test_uncached_call_is_unchanged(self):
        """The common path must not move."""
        result = _extract(
            _usage(
                input_tokens=250,
                output_tokens=100,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
        )
        assert result["input_tokens"] == 250
        assert result["total_tokens"] == 350

    def test_absent_cache_fields_are_treated_as_zero(self):
        """Older SDK responses omit the cache counters entirely."""
        result = _extract(_usage(input_tokens=250, output_tokens=100))
        assert result["total_tokens"] == 350
        assert result["cached_prompt_tokens"] == 0

    def test_none_cache_fields_are_treated_as_zero(self):
        """The SDK sends null rather than 0 when caching is not in play."""
        result = _extract(
            _usage(
                input_tokens=250,
                output_tokens=100,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            )
        )
        assert result["total_tokens"] == 350

    def test_response_without_usage_reports_zero(self):
        """Guard the existing early return rather than assuming it."""
        assert AnthropicCompletion._extract_anthropic_token_usage(
            SimpleNamespace(usage=None)
        ) == {"total_tokens": 0}

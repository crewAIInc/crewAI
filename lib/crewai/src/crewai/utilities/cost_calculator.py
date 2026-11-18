"""Cost calculation utilities for CrewAI spend containment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from crewai.types.usage_metrics import UsageMetrics


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CATALOG_CACHE: dict[str, Any] | None = None

# Fallback token rates when a model is not found in the LiteLLM catalog ($1.50 / $2.00 per 1M tokens)
DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN: float = 1.5e-6
DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN: float = 2.0e-6


def get_model_pricing(model_name: str | None) -> dict[str, float]:
    """Retrieve pricing info for a model from the LiteLLM catalog.

    Returns a dict with 'input_cost_per_token', 'output_cost_per_token',
    and 'cache_read_input_token_cost'.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        try:
            from crewai.llms.context_window import _load_catalog

            _CATALOG_CACHE = _load_catalog()
        except Exception as e:
            logger.debug("Failed to load model pricing catalog: %s", e)
            _CATALOG_CACHE = {}

    if not model_name or not _CATALOG_CACHE:
        return {}

    # 1. Exact match
    if model_name in _CATALOG_CACHE:
        return _extract_pricing(_CATALOG_CACHE[model_name])

    # 2. Provider stripped match (e.g. openai/gpt-4o -> gpt-4o)
    if "/" in model_name:
        _, _, bare = model_name.partition("/")
        if bare in _CATALOG_CACHE:
            return _extract_pricing(_CATALOG_CACHE[bare])

    # 3. Provider prefixed match (e.g. gpt-4o -> openai/gpt-4o)
    for provider in ("openai", "anthropic", "gemini", "azure", "bedrock"):
        prefixed = f"{provider}/{model_name}"
        if prefixed in _CATALOG_CACHE:
            return _extract_pricing(_CATALOG_CACHE[prefixed])

    # 4. Partial suffix match
    for k, v in _CATALOG_CACHE.items():
        if isinstance(v, dict) and (k.endswith(f"/{model_name}") or k == model_name):
            return _extract_pricing(v)

    return {}


def _extract_pricing(props: Any) -> dict[str, float]:
    if not isinstance(props, dict):
        return {}
    input_cost = props.get("input_cost_per_token")
    output_cost = props.get("output_cost_per_token")
    if input_cost is None and output_cost is None:
        return {}
    input_rate = float(input_cost or 0.0)
    output_rate = float(output_cost or 0.0)
    cache_cost = props.get("cache_read_input_token_cost")
    cache_rate = float(cache_cost) if cache_cost is not None else input_rate

    return {
        "input_cost_per_token": input_rate,
        "output_cost_per_token": output_rate,
        "cache_read_input_token_cost": cache_rate,
    }


def calculate_token_cost(
    usage: UsageMetrics | None,
    model: str | None = None,
    custom_pricing: dict[str, float] | None = None,
) -> float:
    """Calculate the USD spend for a UsageMetrics object and model.

    Args:
        usage: UsageMetrics containing token counts.
        model: Optional model name to look up in pricing catalog.
        custom_pricing: Optional override dictionary with pricing rates.

    Returns:
        Estimated or exact cost in USD.
    """
    if usage is None:
        return 0.0

    pricing = custom_pricing or get_model_pricing(model)
    if not pricing:
        input_rate = DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
        output_rate = DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN
        cache_rate = input_rate
    else:
        input_rate = pricing.get(
            "input_cost_per_token", DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
        )
        output_rate = pricing.get(
            "output_cost_per_token", DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN
        )
        cache_rate = pricing.get("cache_read_input_token_cost", input_rate)

    cached_prompt = getattr(usage, "cached_prompt_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    uncached_prompt = max(0, prompt_tokens - cached_prompt)
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    cost = (
        (uncached_prompt * input_rate)
        + (cached_prompt * cache_rate)
        + (completion_tokens * output_rate)
    )
    return max(0.0, float(cost))

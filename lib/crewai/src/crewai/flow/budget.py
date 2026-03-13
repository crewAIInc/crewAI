"""Budget decorator for Flow methods.

This module provides the @budget decorator that enables budget, token, and request
limit enforcement within CrewAI Flows. It allows pausing, stopping, or warning
when token usage, estimated costs, or LLM request counts exceed configured thresholds.

Supports HITL (human-in-the-loop) integration for budget approval when on_exceed='pause'.

Example:
    ```python
    from crewai.flow import Flow, start, listen, budget

    class BudgetedFlow(Flow):
        @start()
        @budget(max_cost=5.00, on_exceed='pause')
        def run_expensive_task(self):
            crew = MyCrew()
            return crew.kickoff()

        @listen(run_expensive_task)
        def process_results(self, result):
            return result
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from crewai.crews.crew_output import CrewOutput
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallStartedEvent
from crewai.types.usage_metrics import UsageMetrics


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import (
        HumanFeedbackProvider,
    )
    from crewai.flow.flow import Flow


logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


# Default model pricing per 1M tokens (input, output)
# Based on approximate pricing as of late 2024
DEFAULT_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # OpenAI models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic models
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    # Google models
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # Fallback for unknown models
    "default": (1.00, 3.00),
}


class BudgetExceededError(Exception):
    """Raised when budget, token, or request limit is exceeded and on_exceed='stop'.

    Attributes:
        current_cost: The current accumulated cost.
        budget_limit: The configured budget limit.
        total_tokens: The total tokens used.
        token_limit: The configured token limit (if any).
        total_requests: The total LLM requests made.
        request_limit: The configured request limit (if any).
        message: Human-readable error message.
    """

    def __init__(
        self,
        current_cost: float,
        budget_limit: float | None = None,
        total_tokens: int = 0,
        token_limit: int | None = None,
        total_requests: int = 0,
        request_limit: int | None = None,
        message: str | None = None,
    ) -> None:
        self.current_cost = current_cost
        self.budget_limit = budget_limit
        self.total_tokens = total_tokens
        self.token_limit = token_limit
        self.total_requests = total_requests
        self.request_limit = request_limit

        if message is None:
            parts = []
            if budget_limit is not None and current_cost >= budget_limit:
                parts.append(f"Budget exceeded: ${current_cost:.2f} >= ${budget_limit:.2f}")
            if token_limit is not None and total_tokens >= token_limit:
                parts.append(f"Token limit exceeded: {total_tokens} >= {token_limit}")
            if request_limit is not None and total_requests >= request_limit:
                parts.append(f"Request limit exceeded: {total_requests} >= {request_limit}")
            message = " | ".join(parts) if parts else "Budget exceeded"

        super().__init__(message)


@dataclass
class BudgetConfig:
    """Configuration for the @budget decorator.

    Attributes:
        max_cost: Maximum allowed cost in USD. None means no budget limit.
        max_tokens: Maximum allowed tokens. None means no token limit.
        max_requests: Maximum allowed LLM requests. None means no request limit.
        on_exceed: Action when limits are exceeded:
            - 'pause': Use HITL to ask for approval (default)
            - 'stop': Raise BudgetExceededError
            - 'warn': Log warning and continue
        cost_per_prompt_token: Custom flat cost per prompt token (overrides cost_map).
        cost_per_completion_token: Custom flat cost per completion token (overrides cost_map).
        cost_map: Custom model pricing (per 1M tokens: input, output).
        provider: Custom HITL provider for 'pause' mode.
    """

    max_cost: float | None = None
    max_tokens: int | None = None
    max_requests: int | None = None
    on_exceed: Literal["pause", "stop", "warn"] = "pause"
    cost_per_prompt_token: float | None = None
    cost_per_completion_token: float | None = None
    cost_map: dict[str, tuple[float, float]] | None = None
    provider: HumanFeedbackProvider | None = None


@dataclass
class BudgetTracker:
    """Tracks cumulative token usage, costs, and requests within a Flow.

    Attributes:
        total_tokens: Total tokens consumed.
        prompt_tokens: Tokens used in prompts.
        completion_tokens: Tokens used in completions.
        successful_requests: Number of successful API requests (from usage metrics).
        total_requests: Total LLM requests tracked via event bus.
        estimated_cost: Estimated cost in USD.
        max_cost: The configured budget limit.
        max_tokens: The configured token limit.
        max_requests: The configured request limit.
        approved_budget: Additional budget approved via HITL.
        approved_tokens: Additional tokens approved via HITL.
        approved_requests: Additional requests approved via HITL.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_requests: int = 0
    estimated_cost: float = 0.0
    max_cost: float | None = None
    max_tokens: int | None = None
    max_requests: int | None = None
    approved_budget: float = 0.0
    approved_tokens: int = 0
    approved_requests: int = 0
    _cost_map: dict[str, tuple[float, float]] = field(default_factory=dict)
    _cost_per_prompt_token: float | None = None
    _cost_per_completion_token: float | None = None

    def __post_init__(self) -> None:
        if not self._cost_map:
            self._cost_map = DEFAULT_MODEL_COSTS.copy()

    def add_usage(
        self,
        usage: UsageMetrics,
        model: str | None = None,
    ) -> None:
        """Add usage metrics and update estimated cost.

        Args:
            usage: The UsageMetrics from a crew/agent execution.
            model: The model name for pricing lookup.
        """
        self.total_tokens += usage.total_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.successful_requests += usage.successful_requests

        # Calculate cost based on prompt/completion tokens
        # If only total_tokens is set (no breakdown), use a 70/30 estimate
        if usage.prompt_tokens == 0 and usage.completion_tokens == 0 and usage.total_tokens > 0:
            # Estimate 70% prompt, 30% completion if no breakdown provided
            estimated_prompt = int(usage.total_tokens * 0.7)
            estimated_completion = usage.total_tokens - estimated_prompt
        else:
            estimated_prompt = usage.prompt_tokens
            estimated_completion = usage.completion_tokens

        # Priority: flat pricing > cost_map > default
        if self._cost_per_prompt_token is not None and self._cost_per_completion_token is not None:
            # Use flat per-token pricing (already per-token, not per-1M)
            prompt_cost_usd = estimated_prompt * self._cost_per_prompt_token
            completion_cost_usd = estimated_completion * self._cost_per_completion_token
        else:
            # Use cost_map pricing (per 1M tokens)
            input_cost, output_cost = self._get_model_pricing(model)
            prompt_cost_usd = (estimated_prompt / 1_000_000) * input_cost
            completion_cost_usd = (estimated_completion / 1_000_000) * output_cost

        self.estimated_cost += prompt_cost_usd + completion_cost_usd

    def increment_request_count(self) -> None:
        """Increment the LLM request counter."""
        self.total_requests += 1

    def _get_model_pricing(self, model: str | None) -> tuple[float, float]:
        """Get pricing for a model (per 1M tokens: input, output)."""
        if model is None:
            return self._cost_map.get("default", (1.00, 3.00))

        # Try exact match first
        if model in self._cost_map:
            return self._cost_map[model]

        # Try prefix matching for model variants, preferring longest match
        model_lower = model.lower()
        best_match: tuple[str, tuple[float, float]] | None = None
        for prefix, pricing in self._cost_map.items():
            if prefix == "default":
                continue
            if model_lower.startswith(prefix.lower()):
                if best_match is None or len(prefix) > len(best_match[0]):
                    best_match = (prefix, pricing)

        if best_match is not None:
            return best_match[1]

        return self._cost_map.get("default", (1.00, 3.00))

    @property
    def effective_budget(self) -> float | None:
        """The total budget including approved additional amounts."""
        if self.max_cost is None:
            return None
        return self.max_cost + self.approved_budget

    @property
    def budget_remaining(self) -> float | None:
        """Remaining budget, or None if no budget limit is set."""
        effective = self.effective_budget
        if effective is None:
            return None
        return max(0.0, effective - self.estimated_cost)

    @property
    def is_budget_exceeded(self) -> bool:
        """Check if the budget has been exceeded."""
        effective = self.effective_budget
        if effective is None:
            return False
        return self.estimated_cost >= effective

    @property
    def effective_token_limit(self) -> int | None:
        """The total token limit including approved additional amounts."""
        if self.max_tokens is None:
            return None
        return self.max_tokens + self.approved_tokens

    @property
    def is_token_limit_exceeded(self) -> bool:
        """Check if the token limit has been exceeded."""
        effective = self.effective_token_limit
        if effective is None:
            return False
        return self.total_tokens >= effective

    @property
    def effective_request_limit(self) -> int | None:
        """The total request limit including approved additional amounts."""
        if self.max_requests is None:
            return None
        return self.max_requests + self.approved_requests

    @property
    def is_request_limit_exceeded(self) -> bool:
        """Check if the request limit has been exceeded."""
        effective = self.effective_request_limit
        if effective is None:
            return False
        return self.total_requests >= effective

    def to_dict(self) -> dict[str, Any]:
        """Return a summary dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "successful_requests": self.successful_requests,
            "total_requests": self.total_requests,
            "estimated_cost": round(self.estimated_cost, 4),
            "max_cost": self.max_cost,
            "max_tokens": self.max_tokens,
            "max_requests": self.max_requests,
            "approved_budget": self.approved_budget,
            "approved_tokens": self.approved_tokens,
            "approved_requests": self.approved_requests,
            "effective_budget": self.effective_budget,
            "effective_token_limit": self.effective_token_limit,
            "effective_request_limit": self.effective_request_limit,
            "budget_remaining": round(self.budget_remaining, 4) if self.budget_remaining is not None else None,
            "is_budget_exceeded": self.is_budget_exceeded,
            "is_token_limit_exceeded": self.is_token_limit_exceeded,
            "is_request_limit_exceeded": self.is_request_limit_exceeded,
        }


def _extract_usage_from_result(result: Any) -> tuple[UsageMetrics | None, str | None]:
    """Extract usage metrics and model from a method result.

    Supports:
    - CrewOutput (from crew.kickoff())
    - LiteAgentOutput (from lite_agent.kickoff())
    - Objects with token_usage or usage_metrics attributes
    - Dicts with token_usage key
    - Lists of the above

    Args:
        result: The return value from a flow method.

    Returns:
        Tuple of (UsageMetrics, model_name) or (None, None) if not found.
    """
    # Handle CrewOutput directly
    if isinstance(result, CrewOutput):
        return result.token_usage, None

    # Handle list of results
    if isinstance(result, list):
        combined = UsageMetrics()
        found_any = False
        for item in result:
            usage, _ = _extract_usage_from_result(item)
            if usage is not None:
                combined.add_usage_metrics(usage)
                found_any = True
        if found_any:
            return combined, None

    # Handle dict with token_usage key
    if isinstance(result, dict):
        if "token_usage" in result:
            usage = result["token_usage"]
            if isinstance(usage, UsageMetrics):
                return usage, result.get("model")
            if isinstance(usage, dict):
                return UsageMetrics(**usage), result.get("model")
        if "usage_metrics" in result:
            usage = result["usage_metrics"]
            if isinstance(usage, UsageMetrics):
                return usage, result.get("model")
            if isinstance(usage, dict):
                return UsageMetrics(**usage), result.get("model")

    # Handle objects with token_usage attribute (CrewOutput, etc.)
    if hasattr(result, "token_usage"):
        usage = result.token_usage
        if isinstance(usage, UsageMetrics):
            return usage, getattr(result, "model", None)

    # Handle LiteAgentOutput with usage_metrics attribute
    if hasattr(result, "usage_metrics"):
        usage_data = result.usage_metrics
        if isinstance(usage_data, UsageMetrics):
            return usage_data, getattr(result, "model", None)
        if isinstance(usage_data, dict):
            return UsageMetrics(**usage_data), getattr(result, "model", None)

    return None, None


def _request_budget_approval(
    flow_instance: Flow[Any],
    config: BudgetConfig,
    tracker: BudgetTracker,
    method_name: str,
) -> str:
    """Request HITL approval to continue with exceeded budget.

    Args:
        flow_instance: The flow instance.
        config: The budget configuration.
        tracker: The current budget tracker.
        method_name: Name of the method that triggered the check.

    Returns:
        The human's feedback string.

    Raises:
        HumanFeedbackPending: If an async provider is used.
    """
    from crewai.flow.async_feedback.types import PendingFeedbackContext
    from crewai.flow.flow_config import flow_config

    # Determine which limit was hit
    limits_hit = []
    if tracker.is_budget_exceeded:
        limits_hit.append(f"cost (${tracker.estimated_cost:.2f} >= ${tracker.max_cost:.2f})")
    if tracker.is_token_limit_exceeded:
        limits_hit.append(f"tokens ({tracker.total_tokens:,} >= {tracker.max_tokens:,})")
    if tracker.is_request_limit_exceeded:
        limits_hit.append(f"requests ({tracker.total_requests} >= {tracker.max_requests})")

    limits_str = ", ".join(limits_hit)

    # Build message for human
    budget_str = f"${tracker.max_cost:.2f}" if tracker.max_cost else "unlimited"
    current_str = f"${tracker.estimated_cost:.2f}"

    message = (
        f"Budget limit reached!\n\n"
        f"Your flow '{flow_instance.name or flow_instance.__class__.__name__}' "
        f"has hit the following limit(s): {limits_str}\n\n"
        f"Current spend breakdown:\n"
        f"  - Estimated cost: {current_str} of {budget_str} budget\n"
        f"  - Tokens used: {tracker.total_tokens:,}\n"
        f"  - LLM requests made: {tracker.total_requests}\n\n"
        f"To continue, approve additional budget/tokens/requests or deny to stop the flow."
    )

    # Build context for provider
    context = PendingFeedbackContext(
        flow_id=flow_instance.flow_id or "unknown",
        flow_class=f"{flow_instance.__class__.__module__}.{flow_instance.__class__.__name__}",
        method_name=method_name,
        method_output=tracker.to_dict(),
        message=message,
        emit=["approved", "denied"],
        default_outcome="denied",
        metadata={
            "budget": True,
            "current_cost": tracker.estimated_cost,
            "max_cost": tracker.max_cost,
            "total_tokens": tracker.total_tokens,
            "max_tokens": tracker.max_tokens,
            "total_requests": tracker.total_requests,
            "max_requests": tracker.max_requests,
        },
        llm=None,
    )

    # Determine effective provider
    effective_provider = config.provider
    if effective_provider is None:
        effective_provider = flow_config.hitl_provider

    if effective_provider is not None:
        return effective_provider.request_feedback(context, flow_instance)

    # Default to console input
    return flow_instance._request_human_feedback(
        message=message,
        output=tracker.to_dict(),
        metadata={"budget": True},
        emit=["approved", "denied"],
    )


def _handle_exceeded_budget(
    flow_instance: Flow[Any],
    config: BudgetConfig,
    tracker: BudgetTracker,
    method_name: str,
) -> None:
    """Handle budget/token/request limit exceeded based on on_exceed setting.

    Args:
        flow_instance: The flow instance.
        config: The budget configuration.
        tracker: The current budget tracker.
        method_name: Name of the method that triggered the check.

    Raises:
        BudgetExceededError: If on_exceed='stop'.
        HumanFeedbackPending: If on_exceed='pause' with async provider.
    """
    if config.on_exceed == "warn":
        logger.warning(
            f"Budget/token/request limit exceeded in flow '{flow_instance.name or flow_instance.__class__.__name__}': "
            f"cost=${tracker.estimated_cost:.2f}, tokens={tracker.total_tokens}, requests={tracker.total_requests}, "
            f"max_cost=${tracker.max_cost}, max_tokens={tracker.max_tokens}, max_requests={tracker.max_requests}"
        )
        return

    if config.on_exceed == "stop":
        raise BudgetExceededError(
            current_cost=tracker.estimated_cost,
            budget_limit=tracker.max_cost,
            total_tokens=tracker.total_tokens,
            token_limit=tracker.max_tokens,
            total_requests=tracker.total_requests,
            request_limit=tracker.max_requests,
        )

    # on_exceed == 'pause' - use HITL
    feedback = _request_budget_approval(flow_instance, config, tracker, method_name)

    # Process feedback to determine approval
    import re

    # Use word-boundary matching to avoid false positives (e.g., "know" contains "no")
    feedback_lower = feedback.lower().strip()
    denial_pattern = r'\b(denied|deny|no|stop|reject)\b'

    if not feedback_lower or re.search(denial_pattern, feedback_lower):
        raise BudgetExceededError(
            current_cost=tracker.estimated_cost,
            budget_limit=tracker.max_cost,
            total_tokens=tracker.total_tokens,
            token_limit=tracker.max_tokens,
            total_requests=tracker.total_requests,
            request_limit=tracker.max_requests,
            message="Budget continuation denied by human reviewer",
        )

    # Approved - increase budget and/or token limit and/or request limit
    # Try to extract numbers from the feedback
    amount_match = re.search(r'\$?(\d+(?:\.\d{1,2})?)', feedback)

    # Handle budget approval
    if tracker.max_cost is not None:
        if amount_match:
            additional_budget = float(amount_match.group(1))
        else:
            # Default: approve same amount as original budget
            additional_budget = tracker.max_cost
        tracker.approved_budget += additional_budget
        logger.info(
            f"Budget approved: +${additional_budget:.2f} "
            f"(new effective budget: ${tracker.effective_budget:.2f})"
        )

    # Handle token limit approval
    if tracker.max_tokens is not None:
        # Try to extract a token count from feedback like "100000 tokens" or just a number
        token_match = re.search(r'(\d+)\s*(?:tokens?|k)?', feedback_lower)
        if token_match:
            additional_tokens = int(token_match.group(1))
            # Handle "100k" style input
            if "k" in feedback_lower[token_match.end()-2:token_match.end()+2]:
                additional_tokens *= 1000
        else:
            # Default: approve same amount as original token limit
            additional_tokens = tracker.max_tokens
        tracker.approved_tokens += additional_tokens
        logger.info(
            f"Token limit approved: +{additional_tokens:,} "
            f"(new effective limit: {tracker.effective_token_limit:,})"
        )

    # Handle request limit approval
    if tracker.max_requests is not None:
        # Try to extract a request count from feedback
        request_match = re.search(r'(\d+)\s*(?:requests?)?', feedback_lower)
        if request_match:
            additional_requests = int(request_match.group(1))
        else:
            # Default: approve same amount as original request limit
            additional_requests = tracker.max_requests
        tracker.approved_requests += additional_requests
        logger.info(
            f"Request limit approved: +{additional_requests} "
            f"(new effective limit: {tracker.effective_request_limit})"
        )


def budget(
    max_cost: float | None = None,
    max_tokens: int | None = None,
    max_requests: int | None = None,
    on_exceed: Literal["pause", "stop", "warn"] = "pause",
    cost_per_prompt_token: float | None = None,
    cost_per_completion_token: float | None = None,
    cost_map: dict[str, tuple[float, float]] | None = None,
    provider: HumanFeedbackProvider | None = None,
) -> Callable[[F], F]:
    """Decorator for Flow methods that enforces budget, token, and request limits.

    This decorator wraps a Flow method to:
    1. Register an event listener to count LLM requests during execution
    2. Execute the method and capture its result
    3. Extract token usage from the result (CrewOutput, LiteAgentOutput, etc.)
    4. Accumulate usage in the flow's budget tracker
    5. Check if any limit (cost, tokens, requests) is exceeded
    6. Take action based on on_exceed setting (pause, stop, or warn)

    When `on_exceed='pause'`, the decorator uses the HITL (human-in-the-loop)
    infrastructure to request budget approval. If approved, the limits are
    increased and execution continues.

    Args:
        max_cost: Maximum allowed cost in USD. None means no budget limit.
        max_tokens: Maximum allowed tokens. None means no token limit.
        max_requests: Maximum allowed LLM requests. None means no request limit.
        on_exceed: Action when limits are exceeded:
            - 'pause': Use HITL to ask for approval to continue (default)
            - 'stop': Raise BudgetExceededError immediately
            - 'warn': Log warning and continue execution
        cost_per_prompt_token: Custom cost per single prompt token (e.g., 0.000003 for $3/1M).
            Takes priority over cost_map when set.
        cost_per_completion_token: Custom cost per single completion token.
            Takes priority over cost_map when set.
        cost_map: Custom model pricing dictionary (per 1M tokens).
            Keys are model name prefixes, values are (input_cost, output_cost).
            Merges with DEFAULT_MODEL_COSTS.
        provider: Custom HumanFeedbackProvider for 'pause' mode.
            If not provided, uses flow_config.hitl_provider or console.

    Returns:
        A decorator function that wraps the method with budget governance.

    Raises:
        BudgetExceededError: If limits exceeded and on_exceed='stop' or denied.
        HumanFeedbackPending: If on_exceed='pause' with async provider.

    Example:
        Basic cost limit with pause on exceed:
        ```python
        @start()
        @budget(max_cost=5.00)
        def expensive_task(self):
            crew = MyCrew()
            return crew.kickoff()
        ```

        Token limit with stop on exceed:
        ```python
        @start()
        @budget(max_tokens=100000, on_exceed='stop')
        def limited_task(self):
            crew = MyCrew()
            return crew.kickoff()
        ```

        Request limit to cap LLM calls:
        ```python
        @start()
        @budget(max_requests=10, on_exceed='stop')
        def limited_requests_task(self):
            # Each LLM call counts against the limit
            agent = MyAgent()
            return agent.kickoff()
        ```

        Combined limits with custom pricing:
        ```python
        @start()
        @budget(
            max_cost=10.00,
            max_tokens=500000,
            max_requests=50,
            on_exceed='pause',
            cost_per_prompt_token=0.000003,  # $3 per 1M tokens
            cost_per_completion_token=0.000015,  # $15 per 1M tokens
        )
        def custom_task(self):
            crew = MyCrew()
            return crew.kickoff()
        ```
    """
    # Validate inputs
    if max_cost is not None and max_cost <= 0:
        raise ValueError("max_cost must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if on_exceed not in ("pause", "stop", "warn"):
        raise ValueError("on_exceed must be 'pause', 'stop', or 'warn'")
    if (cost_per_prompt_token is not None) != (cost_per_completion_token is not None):
        raise ValueError("cost_per_prompt_token and cost_per_completion_token must both be set or both be None")

    # Build config
    config = BudgetConfig(
        max_cost=max_cost,
        max_tokens=max_tokens,
        max_requests=max_requests,
        on_exceed=on_exceed,
        cost_per_prompt_token=cost_per_prompt_token,
        cost_per_completion_token=cost_per_completion_token,
        cost_map=cost_map,
        provider=provider,
    )

    def decorator(func: F) -> F:
        """Inner decorator that wraps the function."""

        def _ensure_budget_tracker(flow_instance: Flow[Any]) -> BudgetTracker:
            """Ensure the flow has a budget tracker initialized."""
            if not hasattr(flow_instance, "_budget_tracker") or flow_instance._budget_tracker is None:
                merged_costs = DEFAULT_MODEL_COSTS.copy()
                if config.cost_map:
                    merged_costs.update(config.cost_map)
                flow_instance._budget_tracker = BudgetTracker(
                    max_cost=config.max_cost,
                    max_tokens=config.max_tokens,
                    max_requests=config.max_requests,
                    _cost_map=merged_costs,
                    _cost_per_prompt_token=config.cost_per_prompt_token,
                    _cost_per_completion_token=config.cost_per_completion_token,
                )
            else:
                # Update limits if not set (in case multiple decorators)
                tracker = flow_instance._budget_tracker
                if tracker.max_cost is None and config.max_cost is not None:
                    tracker.max_cost = config.max_cost
                if tracker.max_tokens is None and config.max_tokens is not None:
                    tracker.max_tokens = config.max_tokens
                if tracker.max_requests is None and config.max_requests is not None:
                    tracker.max_requests = config.max_requests
            return flow_instance._budget_tracker

        def _create_request_counter(tracker: BudgetTracker) -> Callable[[Any, LLMCallStartedEvent], None]:
            """Create an event handler that counts LLM requests."""
            def handler(source: Any, event: LLMCallStartedEvent) -> None:
                tracker.increment_request_count()
            return handler

        def _process_result(
            flow_instance: Flow[Any],
            result: Any,
            method_name: str,
        ) -> Any:
            """Process the method result and check limits."""
            tracker = _ensure_budget_tracker(flow_instance)

            # Extract and accumulate usage
            usage, model = _extract_usage_from_result(result)
            if usage is not None:
                tracker.add_usage(usage, model)

            # Check limits (any one being exceeded triggers the action)
            if tracker.is_budget_exceeded or tracker.is_token_limit_exceeded or tracker.is_request_limit_exceeded:
                _handle_exceeded_budget(flow_instance, config, tracker, method_name)

            return result

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(self: Flow[Any], *args: Any, **kwargs: Any) -> Any:
                tracker = _ensure_budget_tracker(self)
                request_handler = _create_request_counter(tracker)

                # Register event listener for request counting
                crewai_event_bus.register_handler(LLMCallStartedEvent, request_handler)
                try:
                    result = await func(self, *args, **kwargs)
                    return _process_result(self, result, func.__name__)
                finally:
                    # Unregister the event listener
                    crewai_event_bus.off(LLMCallStartedEvent, request_handler)

            wrapper: Any = async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(self: Flow[Any], *args: Any, **kwargs: Any) -> Any:
                tracker = _ensure_budget_tracker(self)
                request_handler = _create_request_counter(tracker)

                # Register event listener for request counting
                crewai_event_bus.register_handler(LLMCallStartedEvent, request_handler)
                try:
                    result = func(self, *args, **kwargs)
                    return _process_result(self, result, func.__name__)
                finally:
                    # Unregister the event listener
                    crewai_event_bus.off(LLMCallStartedEvent, request_handler)

            wrapper = sync_wrapper

        # Preserve existing Flow decorator attributes
        for attr in [
            "__is_start_method__",
            "__trigger_methods__",
            "__condition_type__",
            "__trigger_condition__",
            "__is_flow_method__",
            "__is_router__",
            "__router_paths__",
            "__human_feedback_config__",
        ]:
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))

        # Add budget specific attributes
        wrapper.__budget_config__ = config
        wrapper.__is_flow_method__ = True

        return wrapper  # type: ignore[return-value]

    return decorator


# Backwards compatibility aliases
CostGovernorConfig = BudgetConfig
CostTracker = BudgetTracker
cost_governor = budget

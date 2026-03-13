"""Tests for Flow cost governor functionality."""

import pytest
from unittest.mock import MagicMock, patch

from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow, start, listen
from crewai.flow.cost_governor import (
    BudgetExceededError,
    CostGovernorConfig,
    CostTracker,
    DEFAULT_MODEL_COSTS,
    cost_governor,
    _extract_usage_from_result,
)
from crewai.types.usage_metrics import UsageMetrics


class TestCostTracker:
    """Tests for the CostTracker class."""

    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = CostTracker()
        assert tracker.total_tokens == 0
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.successful_requests == 0
        assert tracker.estimated_cost == 0.0
        assert tracker.budget_limit is None
        assert tracker.token_limit is None
        assert tracker.approved_budget == 0.0

    def test_add_usage(self):
        """Test adding usage metrics."""
        tracker = CostTracker()
        usage = UsageMetrics(
            total_tokens=1000,
            prompt_tokens=700,
            completion_tokens=300,
            successful_requests=1,
        )
        tracker.add_usage(usage)

        assert tracker.total_tokens == 1000
        assert tracker.prompt_tokens == 700
        assert tracker.completion_tokens == 300
        assert tracker.successful_requests == 1
        assert tracker.estimated_cost > 0

    def test_add_usage_with_model_pricing(self):
        """Test adding usage with specific model pricing."""
        tracker = CostTracker()
        # GPT-4o pricing: $2.50/$10.00 per 1M tokens
        usage = UsageMetrics(
            total_tokens=1_000_000,
            prompt_tokens=700_000,
            completion_tokens=300_000,
            successful_requests=1,
        )
        tracker.add_usage(usage, model="gpt-4o")

        # Expected: (700,000 / 1M) * $2.50 + (300,000 / 1M) * $10.00 = $1.75 + $3.00 = $4.75
        assert abs(tracker.estimated_cost - 4.75) < 0.01

    def test_add_usage_accumulates(self):
        """Test that usage accumulates across multiple calls."""
        tracker = CostTracker()

        usage1 = UsageMetrics(total_tokens=500, prompt_tokens=300, completion_tokens=200)
        usage2 = UsageMetrics(total_tokens=500, prompt_tokens=300, completion_tokens=200)

        tracker.add_usage(usage1)
        tracker.add_usage(usage2)

        assert tracker.total_tokens == 1000
        assert tracker.prompt_tokens == 600
        assert tracker.completion_tokens == 400

    def test_budget_exceeded(self):
        """Test budget exceeded detection."""
        tracker = CostTracker(budget_limit=1.00)
        tracker.estimated_cost = 1.50

        assert tracker.is_budget_exceeded is True
        assert tracker.budget_remaining == 0.0

    def test_budget_not_exceeded(self):
        """Test budget not exceeded."""
        tracker = CostTracker(budget_limit=5.00)
        tracker.estimated_cost = 1.00

        assert tracker.is_budget_exceeded is False
        assert tracker.budget_remaining == 4.00

    def test_token_limit_exceeded(self):
        """Test token limit exceeded detection."""
        tracker = CostTracker(token_limit=1000)
        tracker.total_tokens = 1500

        assert tracker.is_token_limit_exceeded is True

    def test_token_limit_not_exceeded(self):
        """Test token limit not exceeded."""
        tracker = CostTracker(token_limit=1000)
        tracker.total_tokens = 500

        assert tracker.is_token_limit_exceeded is False

    def test_effective_budget_with_approved(self):
        """Test effective budget includes approved amounts."""
        tracker = CostTracker(budget_limit=5.00)
        tracker.approved_budget = 3.00

        assert tracker.effective_budget == 8.00

    def test_to_dict(self):
        """Test conversion to dictionary."""
        tracker = CostTracker(budget_limit=5.00, token_limit=1000)
        tracker.total_tokens = 500
        tracker.estimated_cost = 2.00

        result = tracker.to_dict()

        assert result["total_tokens"] == 500
        assert result["estimated_cost"] == 2.00
        assert result["budget_limit"] == 5.00
        assert result["token_limit"] == 1000
        assert result["budget_remaining"] == 3.00
        assert result["is_budget_exceeded"] is False

    def test_model_prefix_matching(self):
        """Test model pricing uses prefix matching."""
        tracker = CostTracker()
        # "gpt-4o-2024-11-20" should match "gpt-4o"
        pricing = tracker._get_model_pricing("gpt-4o-2024-11-20")
        expected = DEFAULT_MODEL_COSTS["gpt-4o"]
        assert pricing == expected

    def test_model_prefix_matching_prefers_longest(self):
        """Test model pricing prefers longest prefix match."""
        tracker = CostTracker()
        # "gpt-4o-mini-2024-07-18" should match "gpt-4o-mini", not "gpt-4o"
        pricing = tracker._get_model_pricing("gpt-4o-mini-2024-07-18")
        expected = DEFAULT_MODEL_COSTS["gpt-4o-mini"]
        assert pricing == expected

        # "o1-mini" should match "o1-mini", not "o1"
        pricing = tracker._get_model_pricing("o1-mini-2024-09-12")
        expected = DEFAULT_MODEL_COSTS["o1-mini"]
        assert pricing == expected

    def test_unknown_model_uses_default(self):
        """Test unknown model uses default pricing."""
        tracker = CostTracker()
        pricing = tracker._get_model_pricing("unknown-model-xyz")
        expected = DEFAULT_MODEL_COSTS["default"]
        assert pricing == expected

    def test_effective_token_limit_with_approved(self):
        """Test effective token limit includes approved amounts."""
        tracker = CostTracker(token_limit=10000)
        tracker.approved_tokens = 5000

        assert tracker.effective_token_limit == 15000

    def test_token_limit_exceeded_considers_approved(self):
        """Test token limit considers approved additional tokens."""
        tracker = CostTracker(token_limit=1000)
        tracker.total_tokens = 1500
        tracker.approved_tokens = 1000

        # 1500 < 2000 (1000 + 1000), so not exceeded
        assert tracker.is_token_limit_exceeded is False


class TestExtractUsageFromResult:
    """Tests for the _extract_usage_from_result function."""

    def test_extract_from_crew_output(self):
        """Test extracting usage from CrewOutput."""
        usage = UsageMetrics(total_tokens=100, prompt_tokens=70, completion_tokens=30)
        crew_output = CrewOutput(raw="test", token_usage=usage)

        result_usage, model = _extract_usage_from_result(crew_output)

        assert result_usage is not None
        assert result_usage.total_tokens == 100

    def test_extract_from_list_of_crew_outputs(self):
        """Test extracting usage from list of CrewOutputs."""
        usage1 = UsageMetrics(total_tokens=100)
        usage2 = UsageMetrics(total_tokens=200)
        crew_outputs = [
            CrewOutput(raw="test1", token_usage=usage1),
            CrewOutput(raw="test2", token_usage=usage2),
        ]

        result_usage, model = _extract_usage_from_result(crew_outputs)

        assert result_usage is not None
        assert result_usage.total_tokens == 300

    def test_extract_from_dict_with_token_usage(self):
        """Test extracting usage from dict with token_usage key."""
        usage = UsageMetrics(total_tokens=100)
        result = {"token_usage": usage, "model": "gpt-4o"}

        result_usage, model = _extract_usage_from_result(result)

        assert result_usage is not None
        assert result_usage.total_tokens == 100
        assert model == "gpt-4o"

    def test_extract_returns_none_for_plain_value(self):
        """Test extracting usage from plain value returns None."""
        result_usage, model = _extract_usage_from_result("plain string")

        assert result_usage is None
        assert model is None

    def test_extract_from_object_with_token_usage_attr(self):
        """Test extracting usage from object with token_usage attribute."""
        class CustomOutput:
            def __init__(self):
                self.token_usage = UsageMetrics(total_tokens=150)
                self.model = "claude-3-sonnet"

        result_usage, model = _extract_usage_from_result(CustomOutput())

        assert result_usage is not None
        assert result_usage.total_tokens == 150


class TestCostGovernorConfig:
    """Tests for the CostGovernorConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CostGovernorConfig()

        assert config.budget_limit is None
        assert config.token_limit is None
        assert config.on_exceed == "pause"
        assert config.cost_map is None
        assert config.provider is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CostGovernorConfig(
            budget_limit=10.00,
            token_limit=50000,
            on_exceed="stop",
            cost_map={"custom-model": (1.0, 2.0)},
        )

        assert config.budget_limit == 10.00
        assert config.token_limit == 50000
        assert config.on_exceed == "stop"
        assert config.cost_map == {"custom-model": (1.0, 2.0)}


class TestBudgetExceededError:
    """Tests for the BudgetExceededError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = BudgetExceededError(current_cost=5.00, budget_limit=3.00)

        assert error.current_cost == 5.00
        assert error.budget_limit == 3.00
        assert "$5.00" in str(error)
        assert "$3.00" in str(error)

    def test_token_limit_error(self):
        """Test error with token limit."""
        error = BudgetExceededError(
            current_cost=0.0,
            total_tokens=15000,
            token_limit=10000,
        )

        assert error.total_tokens == 15000
        assert error.token_limit == 10000
        assert "15000" in str(error)
        assert "10000" in str(error)

    def test_custom_message(self):
        """Test error with custom message."""
        error = BudgetExceededError(
            current_cost=5.00,
            message="Custom error message",
        )

        assert str(error) == "Custom error message"


class TestCostGovernorDecorator:
    """Tests for the @cost_governor decorator."""

    def test_decorator_validation_negative_budget(self):
        """Test decorator raises error for negative budget."""
        with pytest.raises(ValueError, match="budget_limit must be positive"):
            @cost_governor(budget_limit=-1.0)
            def bad_method(self):
                pass

    def test_decorator_validation_negative_tokens(self):
        """Test decorator raises error for negative token limit."""
        with pytest.raises(ValueError, match="token_limit must be positive"):
            @cost_governor(token_limit=-100)
            def bad_method(self):
                pass

    def test_decorator_validation_invalid_on_exceed(self):
        """Test decorator raises error for invalid on_exceed."""
        with pytest.raises(ValueError, match="on_exceed must be"):
            @cost_governor(on_exceed="invalid")
            def bad_method(self):
                pass

    def test_decorator_preserves_flow_attributes(self):
        """Test decorator preserves existing flow method attributes."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=5.00)
            def start_method(self):
                return "result"

        flow = TestFlow()
        method = flow._methods.get("start_method")

        assert method is not None
        assert hasattr(method, "__is_start_method__")
        assert hasattr(method, "__cost_governor_config__")

    def test_flow_initializes_cost_tracker(self):
        """Test flow initializes cost tracker on decorated method execution."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=5.00)
            def start_method(self):
                # Return a CrewOutput to simulate a crew execution
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100, prompt_tokens=70, completion_tokens=30),
                )

        flow = TestFlow()
        flow.kickoff()

        assert flow._cost_tracker is not None
        assert flow._cost_tracker.total_tokens == 100

    def test_cost_accumulates_across_methods(self):
        """Test cost accumulates across multiple decorated methods."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=10.00)
            def step_1(self):
                return CrewOutput(
                    raw="test1",
                    token_usage=UsageMetrics(total_tokens=500),
                )

            @listen(step_1)
            @cost_governor(budget_limit=10.00)
            def step_2(self):
                return CrewOutput(
                    raw="test2",
                    token_usage=UsageMetrics(total_tokens=500),
                )

        flow = TestFlow()
        flow.kickoff()

        assert flow._cost_tracker is not None
        assert flow._cost_tracker.total_tokens == 1000

    def test_cost_summary_accessible(self):
        """Test cost_summary property is accessible."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=5.00)
            def run_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()
        flow.kickoff()

        summary = flow.cost_summary
        assert summary["total_tokens"] == 100
        assert summary["budget_limit"] == 5.00
        assert summary["is_budget_exceeded"] is False

    def test_cost_summary_empty_without_decorator(self):
        """Test cost_summary returns empty dict when no decorator is used."""
        class TestFlow(Flow):
            @start()
            def run_task(self):
                return "no cost tracking"

        flow = TestFlow()
        flow.kickoff()

        summary = flow.cost_summary
        assert summary["total_tokens"] == 0
        assert summary["estimated_cost"] == 0.0
        assert summary["budget_limit"] is None


class TestCostGovernorOnExceedStop:
    """Tests for @cost_governor with on_exceed='stop'."""

    def test_budget_exceeded_raises_error(self):
        """Test budget exceeded raises BudgetExceededError."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="stop")
            def expensive_task(self):
                # This will exceed the tiny budget
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(
                        total_tokens=1_000_000,
                        prompt_tokens=700_000,
                        completion_tokens=300_000,
                    ),
                )

        flow = TestFlow()

        with pytest.raises(BudgetExceededError) as exc_info:
            flow.kickoff()

        assert exc_info.value.budget_limit == 0.001

    def test_token_limit_exceeded_raises_error(self):
        """Test token limit exceeded raises BudgetExceededError."""
        class TestFlow(Flow):
            @start()
            @cost_governor(token_limit=100, on_exceed="stop")
            def limited_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=500),
                )

        flow = TestFlow()

        with pytest.raises(BudgetExceededError) as exc_info:
            flow.kickoff()

        assert exc_info.value.token_limit == 100
        assert exc_info.value.total_tokens == 500


class TestCostGovernorOnExceedWarn:
    """Tests for @cost_governor with on_exceed='warn'."""

    def test_budget_exceeded_logs_warning(self, caplog):
        """Test budget exceeded logs warning and continues."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="warn")
            def step_1(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

            @listen(step_1)
            def step_2(self):
                return "continued"

        flow = TestFlow()
        result = flow.kickoff()

        # Flow should continue despite budget exceeded
        assert result == "continued"

    def test_token_limit_exceeded_logs_warning(self, caplog):
        """Test token limit exceeded logs warning and continues."""
        class TestFlow(Flow):
            @start()
            @cost_governor(token_limit=100, on_exceed="warn")
            def step_1(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=500),
                )

            @listen(step_1)
            def step_2(self):
                return "continued"

        flow = TestFlow()
        result = flow.kickoff()

        assert result == "continued"


class TestCostGovernorOnExceedPause:
    """Tests for @cost_governor with on_exceed='pause' (HITL integration)."""

    def test_budget_exceeded_triggers_pause_denied(self):
        """Test budget exceeded triggers pause and raises on denial."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="pause")
            def expensive_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

        flow = TestFlow()

        # Mock the HITL request to return "denied"
        with patch.object(flow, "_request_human_feedback", return_value="denied"):
            with pytest.raises(BudgetExceededError) as exc_info:
                flow.kickoff()

        assert "denied by human reviewer" in str(exc_info.value)

    def test_budget_exceeded_triggers_pause_approved(self):
        """Test budget exceeded triggers pause and continues on approval."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="pause")
            def step_1(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

            @listen(step_1)
            def step_2(self):
                return "continued after approval"

        flow = TestFlow()

        # Mock the HITL request to return "approved $10.00"
        with patch.object(flow, "_request_human_feedback", return_value="approved $10.00"):
            result = flow.kickoff()

        assert result == "continued after approval"
        assert flow._cost_tracker.approved_budget == 10.00

    def test_budget_exceeded_triggers_pause_approved_without_amount(self):
        """Test approval without explicit amount uses original budget."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=5.00, on_exceed="pause")
            def expensive_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=10_000_000),  # Very expensive
                )

            @listen(expensive_task)
            def step_2(self):
                return "continued"

        flow = TestFlow()

        # Mock the HITL request to return simple "yes, approved"
        with patch.object(flow, "_request_human_feedback", return_value="yes, approved"):
            result = flow.kickoff()

        assert result == "continued"
        # Should add the original budget_limit as approved_budget
        assert flow._cost_tracker.approved_budget == 5.00

    def test_empty_feedback_treated_as_denial(self):
        """Test empty feedback is treated as denial."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="pause")
            def expensive_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value=""):
            with pytest.raises(BudgetExceededError):
                flow.kickoff()


class TestCostGovernorWithCustomCostMap:
    """Tests for @cost_governor with custom cost_map."""

    def test_custom_cost_map_used(self):
        """Test custom cost map is used for pricing."""
        custom_costs = {
            "my-custom-model": (0.50, 1.00),  # $0.50/1M input, $1.00/1M output
        }

        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=100.00, cost_map=custom_costs)
            def run_task(self):
                # Simulate a result with model info
                return {
                    "token_usage": UsageMetrics(
                        total_tokens=2_000_000,
                        prompt_tokens=1_000_000,
                        completion_tokens=1_000_000,
                    ),
                    "model": "my-custom-model",
                }

        flow = TestFlow()
        flow.kickoff()

        # Expected: (1M / 1M) * $0.50 + (1M / 1M) * $1.00 = $1.50
        assert abs(flow._cost_tracker.estimated_cost - 1.50) < 0.01


class TestCostGovernorAsync:
    """Tests for @cost_governor with async flow methods."""

    @pytest.mark.asyncio
    async def test_async_method_with_cost_governor(self):
        """Test cost governor works with async methods."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=5.00)
            async def async_task(self):
                return CrewOutput(
                    raw="async result",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()
        result = await flow.kickoff_async()

        assert flow._cost_tracker is not None
        assert flow._cost_tracker.total_tokens == 100

    @pytest.mark.asyncio
    async def test_async_method_budget_exceeded_stop(self):
        """Test async method with budget exceeded raises error."""
        class TestFlow(Flow):
            @start()
            @cost_governor(budget_limit=0.001, on_exceed="stop")
            async def async_task(self):
                return CrewOutput(
                    raw="expensive",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

        flow = TestFlow()

        with pytest.raises(BudgetExceededError):
            await flow.kickoff_async()


class TestDefaultModelCosts:
    """Tests for the DEFAULT_MODEL_COSTS dictionary."""

    def test_common_models_have_pricing(self):
        """Test common models have pricing defined."""
        expected_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet",
            "claude-3-opus",
            "gemini-1.5-pro",
        ]

        for model in expected_models:
            assert model in DEFAULT_MODEL_COSTS, f"Missing pricing for {model}"
            input_cost, output_cost = DEFAULT_MODEL_COSTS[model]
            assert input_cost > 0
            assert output_cost > 0

    def test_default_fallback_exists(self):
        """Test default fallback pricing exists."""
        assert "default" in DEFAULT_MODEL_COSTS
        input_cost, output_cost = DEFAULT_MODEL_COSTS["default"]
        assert input_cost > 0
        assert output_cost > 0

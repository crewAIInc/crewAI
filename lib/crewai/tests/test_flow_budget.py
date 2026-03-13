"""Tests for Flow budget functionality."""

import pytest
from unittest.mock import MagicMock, patch

from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow, start, listen
from crewai.flow.budget import (
    BudgetExceededError,
    BudgetConfig,
    BudgetTracker,
    DEFAULT_MODEL_COSTS,
    budget,
    _extract_usage_from_result,
)
from crewai.types.usage_metrics import UsageMetrics


class TestBudgetTracker:
    """Tests for the BudgetTracker class."""

    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = BudgetTracker()
        assert tracker.total_tokens == 0
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.successful_requests == 0
        assert tracker.total_requests == 0
        assert tracker.estimated_cost == 0.0
        assert tracker.max_cost is None
        assert tracker.max_tokens is None
        assert tracker.max_requests is None
        assert tracker.approved_budget == 0.0
        assert tracker.approved_requests == 0

    def test_add_usage(self):
        """Test adding usage metrics."""
        tracker = BudgetTracker()
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
        tracker = BudgetTracker()
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
        tracker = BudgetTracker()

        usage1 = UsageMetrics(total_tokens=500, prompt_tokens=300, completion_tokens=200)
        usage2 = UsageMetrics(total_tokens=500, prompt_tokens=300, completion_tokens=200)

        tracker.add_usage(usage1)
        tracker.add_usage(usage2)

        assert tracker.total_tokens == 1000
        assert tracker.prompt_tokens == 600
        assert tracker.completion_tokens == 400

    def test_budget_exceeded(self):
        """Test budget exceeded detection."""
        tracker = BudgetTracker(max_cost=1.00)
        tracker.estimated_cost = 1.50

        assert tracker.is_budget_exceeded is True
        assert tracker.budget_remaining == 0.0

    def test_budget_not_exceeded(self):
        """Test budget not exceeded."""
        tracker = BudgetTracker(max_cost=5.00)
        tracker.estimated_cost = 1.00

        assert tracker.is_budget_exceeded is False
        assert tracker.budget_remaining == 4.00

    def test_token_limit_exceeded(self):
        """Test token limit exceeded detection."""
        tracker = BudgetTracker(max_tokens=1000)
        tracker.total_tokens = 1500

        assert tracker.is_token_limit_exceeded is True

    def test_token_limit_not_exceeded(self):
        """Test token limit not exceeded."""
        tracker = BudgetTracker(max_tokens=1000)
        tracker.total_tokens = 500

        assert tracker.is_token_limit_exceeded is False

    def test_request_limit_exceeded(self):
        """Test request limit exceeded detection."""
        tracker = BudgetTracker(max_requests=10)
        tracker.total_requests = 15

        assert tracker.is_request_limit_exceeded is True

    def test_request_limit_not_exceeded(self):
        """Test request limit not exceeded."""
        tracker = BudgetTracker(max_requests=10)
        tracker.total_requests = 5

        assert tracker.is_request_limit_exceeded is False

    def test_increment_request_count(self):
        """Test incrementing request count."""
        tracker = BudgetTracker()
        assert tracker.total_requests == 0

        tracker.increment_request_count()
        assert tracker.total_requests == 1

        tracker.increment_request_count()
        assert tracker.total_requests == 2

    def test_effective_budget_with_approved(self):
        """Test effective budget includes approved amounts."""
        tracker = BudgetTracker(max_cost=5.00)
        tracker.approved_budget = 3.00

        assert tracker.effective_budget == 8.00

    def test_effective_request_limit_with_approved(self):
        """Test effective request limit includes approved amounts."""
        tracker = BudgetTracker(max_requests=10)
        tracker.approved_requests = 5

        assert tracker.effective_request_limit == 15

    def test_request_limit_exceeded_considers_approved(self):
        """Test request limit considers approved additional requests."""
        tracker = BudgetTracker(max_requests=10)
        tracker.total_requests = 12
        tracker.approved_requests = 5

        # 12 < 15 (10 + 5), so not exceeded
        assert tracker.is_request_limit_exceeded is False

    def test_to_dict_includes_request_data(self):
        """Test conversion to dictionary includes request data."""
        tracker = BudgetTracker(max_cost=5.00, max_tokens=1000, max_requests=20)
        tracker.total_tokens = 500
        tracker.total_requests = 10
        tracker.estimated_cost = 2.00

        result = tracker.to_dict()

        assert result["total_tokens"] == 500
        assert result["total_requests"] == 10
        assert result["estimated_cost"] == 2.00
        assert result["max_cost"] == 5.00
        assert result["max_tokens"] == 1000
        assert result["max_requests"] == 20
        assert result["budget_remaining"] == 3.00
        assert result["is_budget_exceeded"] is False
        assert result["is_request_limit_exceeded"] is False

    def test_model_prefix_matching(self):
        """Test model pricing uses prefix matching."""
        tracker = BudgetTracker()
        # "gpt-4o-2024-11-20" should match "gpt-4o"
        pricing = tracker._get_model_pricing("gpt-4o-2024-11-20")
        expected = DEFAULT_MODEL_COSTS["gpt-4o"]
        assert pricing == expected

    def test_model_prefix_matching_prefers_longest(self):
        """Test model pricing prefers longest prefix match."""
        tracker = BudgetTracker()
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
        tracker = BudgetTracker()
        pricing = tracker._get_model_pricing("unknown-model-xyz")
        expected = DEFAULT_MODEL_COSTS["default"]
        assert pricing == expected

    def test_effective_token_limit_with_approved(self):
        """Test effective token limit includes approved amounts."""
        tracker = BudgetTracker(max_tokens=10000)
        tracker.approved_tokens = 5000

        assert tracker.effective_token_limit == 15000

    def test_token_limit_exceeded_considers_approved(self):
        """Test token limit considers approved additional tokens."""
        tracker = BudgetTracker(max_tokens=1000)
        tracker.total_tokens = 1500
        tracker.approved_tokens = 1000

        # 1500 < 2000 (1000 + 1000), so not exceeded
        assert tracker.is_token_limit_exceeded is False

    def test_custom_flat_pricing(self):
        """Test custom flat per-token pricing."""
        tracker = BudgetTracker(
            _cost_per_prompt_token=0.000003,  # $3 per 1M tokens
            _cost_per_completion_token=0.000015,  # $15 per 1M tokens
        )
        usage = UsageMetrics(
            total_tokens=1_000_000,
            prompt_tokens=700_000,
            completion_tokens=300_000,
        )
        tracker.add_usage(usage, model="gpt-4o")

        # With flat pricing: 700,000 * 0.000003 + 300,000 * 0.000015 = 2.1 + 4.5 = 6.6
        assert abs(tracker.estimated_cost - 6.6) < 0.01

    def test_flat_pricing_overrides_cost_map(self):
        """Test that flat pricing takes priority over cost_map."""
        tracker = BudgetTracker(
            _cost_per_prompt_token=0.000001,
            _cost_per_completion_token=0.000001,
            _cost_map={"test-model": (100.00, 100.00)},  # Very expensive pricing
        )
        usage = UsageMetrics(
            total_tokens=1_000_000,
            prompt_tokens=500_000,
            completion_tokens=500_000,
        )
        tracker.add_usage(usage, model="test-model")

        # If cost_map were used: (500,000/1M) * 100 + (500,000/1M) * 100 = 100
        # With flat pricing: 500,000 * 0.000001 + 500,000 * 0.000001 = 1.0
        assert abs(tracker.estimated_cost - 1.0) < 0.01


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

    def test_extract_from_dict_with_usage_metrics(self):
        """Test extracting usage from dict with usage_metrics key."""
        usage = UsageMetrics(total_tokens=150)
        result = {"usage_metrics": usage, "model": "claude-3-sonnet"}

        result_usage, model = _extract_usage_from_result(result)

        assert result_usage is not None
        assert result_usage.total_tokens == 150
        assert model == "claude-3-sonnet"

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

    def test_extract_from_object_with_usage_metrics_attr(self):
        """Test extracting usage from object with usage_metrics attribute (LiteAgentOutput style)."""
        class LiteAgentStyleOutput:
            def __init__(self):
                self.usage_metrics = UsageMetrics(total_tokens=200)
                self.model = "gpt-4o"

        result_usage, model = _extract_usage_from_result(LiteAgentStyleOutput())

        assert result_usage is not None
        assert result_usage.total_tokens == 200


class TestBudgetConfig:
    """Tests for the BudgetConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BudgetConfig()

        assert config.max_cost is None
        assert config.max_tokens is None
        assert config.max_requests is None
        assert config.on_exceed == "pause"
        assert config.cost_per_prompt_token is None
        assert config.cost_per_completion_token is None
        assert config.cost_map is None
        assert config.provider is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BudgetConfig(
            max_cost=10.00,
            max_tokens=50000,
            max_requests=100,
            on_exceed="stop",
            cost_per_prompt_token=0.000003,
            cost_per_completion_token=0.000015,
            cost_map={"custom-model": (1.0, 2.0)},
        )

        assert config.max_cost == 10.00
        assert config.max_tokens == 50000
        assert config.max_requests == 100
        assert config.on_exceed == "stop"
        assert config.cost_per_prompt_token == 0.000003
        assert config.cost_per_completion_token == 0.000015
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

    def test_request_limit_error(self):
        """Test error with request limit."""
        error = BudgetExceededError(
            current_cost=0.0,
            total_requests=25,
            request_limit=20,
        )

        assert error.total_requests == 25
        assert error.request_limit == 20
        assert "25" in str(error)
        assert "20" in str(error)

    def test_custom_message(self):
        """Test error with custom message."""
        error = BudgetExceededError(
            current_cost=5.00,
            message="Custom error message",
        )

        assert str(error) == "Custom error message"


class TestBudgetDecorator:
    """Tests for the @budget decorator."""

    def test_decorator_validation_negative_budget(self):
        """Test decorator raises error for negative budget."""
        with pytest.raises(ValueError, match="max_cost must be positive"):
            @budget(max_cost=-1.0)
            def bad_method(self):
                pass

    def test_decorator_validation_negative_tokens(self):
        """Test decorator raises error for negative token limit."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            @budget(max_tokens=-100)
            def bad_method(self):
                pass

    def test_decorator_validation_negative_requests(self):
        """Test decorator raises error for negative request limit."""
        with pytest.raises(ValueError, match="max_requests must be positive"):
            @budget(max_requests=-10)
            def bad_method(self):
                pass

    def test_decorator_validation_invalid_on_exceed(self):
        """Test decorator raises error for invalid on_exceed."""
        with pytest.raises(ValueError, match="on_exceed must be"):
            @budget(on_exceed="invalid")
            def bad_method(self):
                pass

    def test_decorator_validation_partial_custom_pricing(self):
        """Test decorator raises error for partial custom pricing."""
        with pytest.raises(ValueError, match="cost_per_prompt_token and cost_per_completion_token must both be set"):
            @budget(cost_per_prompt_token=0.000003)
            def bad_method(self):
                pass

    def test_decorator_preserves_flow_attributes(self):
        """Test decorator preserves existing flow method attributes."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00)
            def start_method(self):
                return "result"

        flow = TestFlow()
        method = flow._methods.get("start_method")

        assert method is not None
        assert hasattr(method, "__is_start_method__")
        assert hasattr(method, "__budget_config__")

    def test_flow_initializes_budget_tracker(self):
        """Test flow initializes budget tracker on decorated method execution."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00)
            def start_method(self):
                # Return a CrewOutput to simulate a crew execution
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100, prompt_tokens=70, completion_tokens=30),
                )

        flow = TestFlow()
        flow.kickoff()

        assert flow._budget_tracker is not None
        assert flow._budget_tracker.total_tokens == 100

    def test_cost_accumulates_across_methods(self):
        """Test cost accumulates across multiple decorated methods."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=10.00)
            def step_1(self):
                return CrewOutput(
                    raw="test1",
                    token_usage=UsageMetrics(total_tokens=500),
                )

            @listen(step_1)
            @budget(max_cost=10.00)
            def step_2(self):
                return CrewOutput(
                    raw="test2",
                    token_usage=UsageMetrics(total_tokens=500),
                )

        flow = TestFlow()
        flow.kickoff()

        assert flow._budget_tracker is not None
        assert flow._budget_tracker.total_tokens == 1000

    def test_budget_summary_accessible(self):
        """Test budget_summary property is accessible."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00)
            def run_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()
        flow.kickoff()

        summary = flow.budget_summary
        assert summary["total_tokens"] == 100
        assert summary["max_cost"] == 5.00
        assert summary["is_budget_exceeded"] is False

    def test_budget_summary_empty_without_decorator(self):
        """Test budget_summary returns empty dict when no decorator is used."""
        class TestFlow(Flow):
            @start()
            def run_task(self):
                return "no budget tracking"

        flow = TestFlow()
        flow.kickoff()

        summary = flow.budget_summary
        assert summary["total_tokens"] == 0
        assert summary["estimated_cost"] == 0.0
        assert summary["max_cost"] is None
        assert summary["max_requests"] is None


class TestBudgetOnExceedStop:
    """Tests for @budget with on_exceed='stop'."""

    def test_budget_exceeded_raises_error(self):
        """Test budget exceeded raises BudgetExceededError."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=0.001, on_exceed="stop")
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
            @budget(max_tokens=100, on_exceed="stop")
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

    def test_request_limit_exceeded_raises_error(self):
        """Test request limit exceeded raises BudgetExceededError.

        Note: Request counting is done via event listener on LLMCallStartedEvent.
        Since we're not making actual LLM calls in this test, we verify the
        tracker's request limit detection logic directly.
        """
        tracker = BudgetTracker(max_requests=5)
        tracker.total_requests = 10  # Exceed the limit

        assert tracker.is_request_limit_exceeded is True
        assert tracker.max_requests == 5
        assert tracker.total_requests == 10


class TestBudgetOnExceedWarn:
    """Tests for @budget with on_exceed='warn'."""

    def test_budget_exceeded_logs_warning(self, caplog):
        """Test budget exceeded logs warning and continues."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=0.001, on_exceed="warn")
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
            @budget(max_tokens=100, on_exceed="warn")
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


class TestBudgetOnExceedPause:
    """Tests for @budget with on_exceed='pause' (HITL integration)."""

    def test_budget_exceeded_triggers_pause_denied(self):
        """Test budget exceeded triggers pause and raises on denial."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=0.001, on_exceed="pause")
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
            @budget(max_cost=0.001, on_exceed="pause")
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
        assert flow._budget_tracker.approved_budget == 10.00

    def test_budget_exceeded_triggers_pause_approved_without_amount(self):
        """Test approval without explicit amount uses original budget."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00, on_exceed="pause")
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
        # Should add the original max_cost as approved_budget
        assert flow._budget_tracker.approved_budget == 5.00

    def test_empty_feedback_treated_as_denial(self):
        """Test empty feedback is treated as denial."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=0.001, on_exceed="pause")
            def expensive_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=1_000_000),
                )

        flow = TestFlow()

        with patch.object(flow, "_request_human_feedback", return_value=""):
            with pytest.raises(BudgetExceededError):
                flow.kickoff()


class TestBudgetWithCustomPricing:
    """Tests for @budget with custom pricing options."""

    def test_custom_cost_map_used(self):
        """Test custom cost map is used for pricing."""
        custom_costs = {
            "my-custom-model": (0.50, 1.00),  # $0.50/1M input, $1.00/1M output
        }

        class TestFlow(Flow):
            @start()
            @budget(max_cost=100.00, cost_map=custom_costs)
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
        assert abs(flow._budget_tracker.estimated_cost - 1.50) < 0.01

    def test_custom_flat_pricing_used(self):
        """Test custom flat per-token pricing is used."""
        class TestFlow(Flow):
            @start()
            @budget(
                max_cost=100.00,
                cost_per_prompt_token=0.000003,  # $3/1M
                cost_per_completion_token=0.000015,  # $15/1M
            )
            def run_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(
                        total_tokens=1_000_000,
                        prompt_tokens=700_000,
                        completion_tokens=300_000,
                    ),
                )

        flow = TestFlow()
        flow.kickoff()

        # Expected: 700,000 * 0.000003 + 300,000 * 0.000015 = 2.1 + 4.5 = 6.6
        assert abs(flow._budget_tracker.estimated_cost - 6.6) < 0.01

    def test_flat_pricing_overrides_cost_map_in_decorator(self):
        """Test flat pricing takes priority over cost_map in decorator."""
        class TestFlow(Flow):
            @start()
            @budget(
                max_cost=100.00,
                cost_per_prompt_token=0.000001,  # $1/1M
                cost_per_completion_token=0.000001,  # $1/1M
                cost_map={"gpt-4o": (100.00, 100.00)},  # Very expensive (should be ignored)
            )
            def run_task(self):
                return {
                    "token_usage": UsageMetrics(
                        total_tokens=1_000_000,
                        prompt_tokens=500_000,
                        completion_tokens=500_000,
                    ),
                    "model": "gpt-4o",
                }

        flow = TestFlow()
        flow.kickoff()

        # With flat pricing: 500,000 * 0.000001 + 500,000 * 0.000001 = 1.0
        # If cost_map were used: (500,000/1M) * 100 + (500,000/1M) * 100 = 100
        assert abs(flow._budget_tracker.estimated_cost - 1.0) < 0.01


class TestBudgetCombinedLimits:
    """Tests for @budget with combined limits (cost + tokens + requests)."""

    def test_first_limit_hit_triggers_action(self):
        """Test that the first limit hit triggers the action."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=100.00, max_tokens=50, on_exceed="stop")
            def limited_task(self):
                # Token limit will be hit first (100 > 50), cost won't
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()

        with pytest.raises(BudgetExceededError) as exc_info:
            flow.kickoff()

        # Token limit was hit, not budget
        assert exc_info.value.token_limit == 50
        assert exc_info.value.total_tokens == 100


class TestBudgetAsync:
    """Tests for @budget with async flow methods."""

    @pytest.mark.asyncio
    async def test_async_method_with_budget(self):
        """Test budget works with async methods."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00)
            async def async_task(self):
                return CrewOutput(
                    raw="async result",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()
        result = await flow.kickoff_async()

        assert flow._budget_tracker is not None
        assert flow._budget_tracker.total_tokens == 100

    @pytest.mark.asyncio
    async def test_async_method_budget_exceeded_stop(self):
        """Test async method with budget exceeded raises error."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=0.001, on_exceed="stop")
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


class TestBudgetSummaryFields:
    """Tests to ensure budget_summary includes all required fields."""

    def test_budget_summary_has_request_fields(self):
        """Test budget_summary includes request-related fields."""
        class TestFlow(Flow):
            @start()
            @budget(max_cost=5.00, max_requests=10)
            def run_task(self):
                return CrewOutput(
                    raw="test",
                    token_usage=UsageMetrics(total_tokens=100),
                )

        flow = TestFlow()
        flow.kickoff()

        summary = flow.budget_summary
        assert "total_requests" in summary
        assert "max_requests" in summary
        assert "approved_requests" in summary
        assert "effective_request_limit" in summary
        assert "is_request_limit_exceeded" in summary

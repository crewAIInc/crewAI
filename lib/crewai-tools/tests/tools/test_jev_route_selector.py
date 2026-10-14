"""Behavioral tests for the Jev Choice confidence-aware route selector."""

from __future__ import annotations

import math
from typing import Any
import pytest

try:
    from crewai_tools.tools.jev_tool.route_selector import (
        select_choice_route,
        select_jev_route,
    )
except ImportError:
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "crewai_tools"
        / "tools"
        / "jev_tool"
        / "route_selector.py"
    )
    spec = importlib.util.spec_from_file_location("route_selector", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    select_jev_route = module.select_jev_route
    select_choice_route = module.select_choice_route

try:
    from crewai.flow.dsl import listen, router, start
    from crewai.flow.flow import Flow

    HAS_FLOW = True
except ImportError:
    HAS_FLOW = False


ROUTES = ["ROUTE_A", "ROUTE_B", "FALLBACK_ROUTE"]
DEFAULT_FALLBACK = "FALLBACK_ROUTE"
DEFAULT_THRESHOLD = 0.70


class TestConfidenceThresholdBehavior:
    """Tests for confidence values relative to the configured threshold."""

    def test_confidence_above_threshold_selects_route(self) -> None:
        """Verify that confidence strictly above the threshold selects the choice route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=0.85,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == "ROUTE_A"

    def test_confidence_equal_to_threshold_selects_route(self) -> None:
        """Verify that confidence exactly equal to the threshold selects the choice route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=DEFAULT_THRESHOLD,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == "ROUTE_A"

    def test_confidence_below_threshold_selects_fallback(self) -> None:
        """Verify that confidence below the threshold selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=0.699,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_exact_boundary_zero_and_one(self) -> None:
        """Verify exact boundary values 0.0 and 1.0 for threshold and confidence."""
        assert (
            select_jev_route("ROUTE_A", 0.0, ROUTES, min_confidence=0.0, fallback=DEFAULT_FALLBACK)
            == "ROUTE_A"
        )
        assert (
            select_jev_route("ROUTE_A", 1.0, ROUTES, min_confidence=1.0, fallback=DEFAULT_FALLBACK)
            == "ROUTE_A"
        )
        assert (
            select_jev_route("ROUTE_A", 0.999, ROUTES, min_confidence=1.0, fallback=DEFAULT_FALLBACK)
            == DEFAULT_FALLBACK
        )


class TestChoiceLabelHandling:
    """Tests for known, unknown, and missing choice labels."""

    def test_known_route_selected_when_confidence_sufficient(self) -> None:
        """Verify known route is selected when confidence meets threshold."""
        assert (
            select_jev_route("ROUTE_B", 0.90, ROUTES, DEFAULT_THRESHOLD, DEFAULT_FALLBACK)
            == "ROUTE_B"
        )

    def test_unknown_choice_label_selects_fallback(self) -> None:
        """Verify unknown choice label selects the fallback route."""
        result = select_jev_route(
            choice="UNKNOWN_ROUTE",
            confidence=0.99,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_none_choice_selects_fallback(self) -> None:
        """Verify None choice selects the fallback route."""
        result = select_jev_route(
            choice=None,
            confidence=0.95,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_empty_string_choice_selects_fallback(self) -> None:
        """Verify empty string choice selects the fallback route."""
        result = select_jev_route(
            choice="",
            confidence=0.95,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_non_string_choice_selects_fallback(self) -> None:
        """Verify non-string choice types select the fallback route."""
        result = select_jev_route(
            choice=123,  # type: ignore[arg-type]
            confidence=0.95,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK


class TestConfidenceValidation:
    """Tests for missing, non-finite, out-of-range, and invalid confidence values."""

    def test_none_confidence_selects_fallback(self) -> None:
        """Verify missing (None) confidence selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=None,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_nan_confidence_selects_fallback(self) -> None:
        """Verify NaN confidence selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=float("nan"),
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_positive_infinity_confidence_selects_fallback(self) -> None:
        """Verify positive infinity confidence selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=float("inf"),
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_negative_infinity_confidence_selects_fallback(self) -> None:
        """Verify negative infinity confidence selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=float("-inf"),
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_negative_out_of_range_confidence_selects_fallback(self) -> None:
        """Verify negative out-of-range confidence selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=-0.01,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_greater_than_one_out_of_range_confidence_selects_fallback(self) -> None:
        """Verify confidence exceeding 1.0 selects the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=1.05,
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_non_numeric_confidence_type_selects_fallback(self) -> None:
        """Verify non-numeric confidence types select the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence="high",  # type: ignore[arg-type]
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK

    def test_boolean_confidence_selects_fallback(self) -> None:
        """Verify boolean confidence values select the fallback route."""
        result = select_jev_route(
            choice="ROUTE_A",
            confidence=True,  # type: ignore[arg-type]
            allowed_routes=ROUTES,
            min_confidence=DEFAULT_THRESHOLD,
            fallback=DEFAULT_FALLBACK,
        )
        assert result == DEFAULT_FALLBACK


class TestInvalidConfiguration:
    """Tests for invalid caller-provided configuration (threshold, fallback, routes)."""

    def test_negative_threshold_raises_value_error(self) -> None:
        """Verify negative min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=-0.1, fallback=DEFAULT_FALLBACK)

    def test_threshold_above_one_raises_value_error(self) -> None:
        """Verify min_confidence greater than 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=1.1, fallback=DEFAULT_FALLBACK)

    def test_nan_threshold_raises_value_error(self) -> None:
        """Verify NaN min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be a finite number"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=float("nan"), fallback=DEFAULT_FALLBACK)

    def test_infinite_threshold_raises_value_error(self) -> None:
        """Verify infinite min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be a finite number"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=float("inf"), fallback=DEFAULT_FALLBACK)

    def test_non_numeric_threshold_raises_type_error(self) -> None:
        """Verify non-numeric min_confidence raises TypeError."""
        with pytest.raises(TypeError, match="min_confidence must be a float or int"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence="high", fallback=DEFAULT_FALLBACK)  # type: ignore[arg-type]

    def test_boolean_threshold_raises_type_error(self) -> None:
        """Verify boolean min_confidence raises TypeError."""
        with pytest.raises(TypeError, match="min_confidence must be a float or int"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=True, fallback=DEFAULT_FALLBACK)  # type: ignore[arg-type]

    def test_fallback_not_in_allowed_routes_raises_value_error(self) -> None:
        """Verify fallback route not present in allowed_routes raises ValueError."""
        with pytest.raises(ValueError, match="fallback route 'INVALID_FALLBACK' must be present"):
            select_jev_route(
                "ROUTE_A",
                0.8,
                ROUTES,
                min_confidence=0.5,
                fallback="INVALID_FALLBACK",
            )

    def test_non_string_fallback_raises_type_error(self) -> None:
        """Verify non-string fallback raises TypeError."""
        with pytest.raises(TypeError, match="fallback must be a string"):
            select_jev_route("ROUTE_A", 0.8, ROUTES, min_confidence=0.5, fallback=123)  # type: ignore[arg-type]

    def test_empty_allowed_routes_raises_value_error(self) -> None:
        """Verify empty allowed_routes raises ValueError."""
        with pytest.raises(ValueError, match="allowed_routes cannot be empty"):
            select_jev_route("ROUTE_A", 0.8, [], min_confidence=0.5, fallback=DEFAULT_FALLBACK)

    def test_non_collection_allowed_routes_raises_type_error(self) -> None:
        """Verify non-collection allowed_routes raises TypeError."""
        with pytest.raises(TypeError, match="allowed_routes must be a collection"):
            select_jev_route("ROUTE_A", 0.8, 123, min_confidence=0.5, fallback=DEFAULT_FALLBACK)  # type: ignore[arg-type]

    def test_single_string_allowed_routes_raises_type_error(self) -> None:
        """Verify single string allowed_routes raises TypeError."""
        with pytest.raises(TypeError, match="allowed_routes must be a collection"):
            select_jev_route("ROUTE_A", 0.8, "ROUTE_A", min_confidence=0.5, fallback="ROUTE_A")  # type: ignore[arg-type]

    def test_allowed_routes_with_empty_or_non_string_raises_value_error(self) -> None:
        """Verify allowed_routes with empty or non-string items raises ValueError."""
        with pytest.raises(ValueError, match="allowed_routes elements must be non-empty strings"):
            select_jev_route("ROUTE_A", 0.8, ["ROUTE_A", ""], min_confidence=0.5, fallback="ROUTE_A")

        with pytest.raises(ValueError, match="allowed_routes elements must be non-empty strings"):
            select_jev_route("ROUTE_A", 0.8, ["ROUTE_A", None], min_confidence=0.5, fallback="ROUTE_A")  # type: ignore[list-item]

    def test_allowed_routes_with_unhashable_element_raises_value_error(self) -> None:
        """Verify allowed_routes with unhashable element raises ValueError."""
        with pytest.raises(ValueError, match="allowed_routes elements must be non-empty strings"):
            select_jev_route("ROUTE_A", 0.8, [["ROUTE_A"]], min_confidence=0.5, fallback="ROUTE_A")  # type: ignore[list-item]


class TestAliasAndExports:
    """Verify alias parity and public exports."""

    def test_select_choice_route_alias_behaves_identically(self) -> None:
        """Verify select_choice_route alias behaves identically to select_jev_route."""
        assert select_choice_route is select_jev_route

        result = select_choice_route("ROUTE_A", 0.85, ROUTES, 0.7, DEFAULT_FALLBACK)
        assert result == "ROUTE_A"

    def test_public_namespace_exports(self) -> None:
        """Verify select_jev_route and select_choice_route are exposed in crewai_tools."""
        try:
            import sys
            from pathlib import Path

            src_dir = str(Path(__file__).resolve().parent.parent.parent / "src")
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            import crewai_tools

            assert hasattr(crewai_tools, "select_jev_route")
            assert hasattr(crewai_tools, "select_choice_route")
            assert crewai_tools.select_jev_route is select_jev_route
            assert crewai_tools.select_choice_route is select_choice_route
        except ImportError:
            # Full crewai-tools dependency tree not present in stripped local environment
            pass


class TestFlowRouterIntegration:
    """Verifies that select_jev_route integrates with Flow routing patterns."""

    def test_router_function_with_synthetic_decision(self) -> None:
        """Verify routing helper within a simulated flow router function."""
        def router_step(state: dict[str, Any]) -> str:
            """Simulate a flow router step extracting decision state."""
            decision = state.get("decision", {})
            return select_jev_route(
                choice=decision.get("choice"),
                confidence=decision.get("confidence"),
                allowed_routes=ROUTES,
                min_confidence=0.8,
                fallback=DEFAULT_FALLBACK,
            )

        assert router_step({"decision": {"choice": "ROUTE_A", "confidence": 0.85}}) == "ROUTE_A"
        assert router_step({"decision": {"choice": "ROUTE_A", "confidence": 0.75}}) == DEFAULT_FALLBACK
        assert router_step({"decision": {"choice": "UNKNOWN", "confidence": 0.99}}) == DEFAULT_FALLBACK
        assert router_step({"decision": {"choice": None, "confidence": 0.95}}) == DEFAULT_FALLBACK

    @pytest.mark.skipif(not HAS_FLOW, reason="crewai Flow requires full workspace dependencies")
    def test_flow_router_routes_to_matched_step(self) -> None:
        """Verify integration with Flow router when decision meets confidence threshold."""
        class SyntheticFlow(Flow):
            """Flow for testing high confidence routing."""

            @start()
            def make_decision(self) -> dict[str, Any]:
                """Generate synthetic decision at flow start."""
                decision = {"choice": "ROUTE_A", "confidence": 0.92}
                self.state["decision"] = decision
                return decision

            @router(make_decision)
            def decide_next(self, res: dict[str, Any] | None = None) -> str:
                """Route flow based on synthetic decision confidence."""
                decision = res or self.state.get("decision", {})
                return select_jev_route(
                    choice=decision.get("choice"),
                    confidence=decision.get("confidence"),
                    allowed_routes=ROUTES,
                    min_confidence=0.8,
                    fallback=DEFAULT_FALLBACK,
                )

            @listen("ROUTE_A")
            def step_a(self) -> str:
                """Handle ROUTE_A execution."""
                return "hit_a"

            @listen(DEFAULT_FALLBACK)
            def step_fallback(self) -> str:
                """Handle fallback execution."""
                return "hit_fallback"

        flow = SyntheticFlow()
        flow_result = flow.kickoff()
        assert flow_result == "hit_a"

    @pytest.mark.skipif(not HAS_FLOW, reason="crewai Flow requires full workspace dependencies")
    def test_flow_router_routes_to_fallback_on_low_confidence(self) -> None:
        """Verify integration with Flow router when decision has low confidence."""
        class SyntheticLowConfidenceFlow(Flow):
            """Flow for testing low confidence fallback routing."""

            @start()
            def make_decision(self) -> dict[str, Any]:
                """Generate synthetic low-confidence decision at flow start."""
                decision = {"choice": "ROUTE_A", "confidence": 0.45}
                self.state["decision"] = decision
                return decision

            @router(make_decision)
            def decide_next(self, res: dict[str, Any] | None = None) -> str:
                """Route flow to fallback based on low confidence."""
                decision = res or self.state.get("decision", {})
                return select_jev_route(
                    choice=decision.get("choice"),
                    confidence=decision.get("confidence"),
                    allowed_routes=ROUTES,
                    min_confidence=0.8,
                    fallback=DEFAULT_FALLBACK,
                )

            @listen("ROUTE_A")
            def step_a(self) -> str:
                """Handle ROUTE_A execution."""
                return "hit_a"

            @listen(DEFAULT_FALLBACK)
            def step_fallback(self) -> str:
                """Handle fallback execution."""
                return "hit_fallback"

        flow = SyntheticLowConfidenceFlow()
        flow_result = flow.kickoff()
        assert flow_result == "hit_fallback"

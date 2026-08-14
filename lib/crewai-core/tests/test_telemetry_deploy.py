"""Deployment telemetry: attribution by origin, and an origin-independent count.

``Create Crew Deployment`` and ``Start Deployment`` answer "which deployment,
from where"; ``deploy:created`` / ``deploy:pushed`` answer "how many
deployments", from the feature-usage aggregation, regardless of origin.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

from crewai_core.telemetry import Telemetry
import pytest


@pytest.fixture
def telemetry() -> Iterator[tuple[Telemetry, MagicMock]]:
    """A Telemetry whose spans are captured instead of exported.

    The singleton is disabled in tests and never builds a provider, so both the
    gate and the provider are supplied here.
    """
    instance = Telemetry()
    span = MagicMock()
    provider = MagicMock()
    provider.get_tracer.return_value.start_span.return_value = span

    with (
        patch.object(instance, "provider", provider, create=True),
        patch.object(instance, "_should_execute_telemetry", return_value=True),
        patch("crewai_core.telemetry.close_span"),
    ):
        yield instance, span


def _attributes(span: MagicMock) -> dict[str, Any]:
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def _span_names(provider: MagicMock) -> list[str]:
    tracer = provider.get_tracer.return_value
    return [call.args[0] for call in tracer.start_span.call_args_list]


class TestCreateDeployment:
    def test_defaults_to_cli(self, telemetry: tuple[Telemetry, MagicMock]) -> None:
        instance, span = telemetry
        instance.create_crew_deployment_span()
        assert _attributes(span)["source"] == "cli"

    def test_records_tui_when_started_from_the_run_ui(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        instance, span = telemetry
        instance.create_crew_deployment_span(source="tui")
        assert _attributes(span)["source"] == "tui"

    def test_also_counts_the_deployment_as_a_feature(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        instance, _ = telemetry
        with patch.object(instance, "feature_usage_span") as feature:
            instance.create_crew_deployment_span()
        feature.assert_called_once_with("deploy:created")

    def test_feature_count_is_the_same_from_either_origin(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        """The whole point: one number for deployments, whatever started them."""
        instance, _ = telemetry
        with patch.object(instance, "feature_usage_span") as feature:
            instance.create_crew_deployment_span(source="cli")
            instance.create_crew_deployment_span(source="tui")
        assert [call.args[0] for call in feature.call_args_list] == [
            "deploy:created",
            "deploy:created",
        ]


class TestStartDeployment:
    def test_defaults_to_cli_and_keeps_the_uuid(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        instance, span = telemetry
        instance.start_deployment_span("dep-123")
        attributes = _attributes(span)
        assert attributes["source"] == "cli"
        assert attributes["uuid"] == "dep-123"

    def test_records_tui_when_started_from_the_run_ui(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        instance, span = telemetry
        instance.start_deployment_span("dep-123", source="tui")
        assert _attributes(span)["source"] == "tui"

    def test_source_is_recorded_even_without_a_uuid(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        """uuid is optional; source must not be conditional on it."""
        instance, span = telemetry
        instance.start_deployment_span(None, source="tui")
        attributes = _attributes(span)
        assert attributes["source"] == "tui"
        assert "uuid" not in attributes

    def test_also_counts_the_deployment_as_a_feature(
        self, telemetry: tuple[Telemetry, MagicMock]
    ) -> None:
        instance, _ = telemetry
        with patch.object(instance, "feature_usage_span") as feature:
            instance.start_deployment_span("dep-123")
        feature.assert_called_once_with("deploy:pushed")


class TestDisabledTelemetry:
    def test_opted_out_users_emit_nothing(self) -> None:
        """No span and no feature count when telemetry is off."""
        instance = Telemetry()
        provider = MagicMock()

        with (
            patch.object(instance, "provider", provider, create=True),
            patch.object(instance, "_should_execute_telemetry", return_value=False),
            patch.object(instance, "feature_usage_span") as feature,
        ):
            instance.create_crew_deployment_span(source="tui")
            instance.start_deployment_span("dep-123", source="tui")

        assert _span_names(provider) == []
        # feature_usage_span is itself gated, so it is still called; it is the
        # export that must not happen. Assert it was not bypassed some other way.
        assert [call.args[0] for call in feature.call_args_list] == [
            "deploy:created",
            "deploy:pushed",
        ]


class TestReleaseAttribution:
    """Every span this emitter produces must carry the running release.

    A span without ``crewai_version`` cannot be attributed to a version, so a
    version-filtered question returns nothing for it rather than something
    visibly wrong. Covers all eight spans this module emits, not only the ones
    that were missing it, so a regression on the others is caught too.
    """

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("deploy_signup_error_span", ()),
            ("start_deployment_span", ("dep-123",)),
            ("create_crew_deployment_span", ()),
            ("get_crew_logs_span", ("dep-123", "deployment")),
            ("remove_crew_span", ("dep-123",)),
            ("feature_usage_span", ("memory:query",)),
            ("flow_creation_span", ("ResearchFlow",)),
            ("template_installed_span", ("my-template",)),
        ],
    )
    def test_span_records_the_release(
        self,
        telemetry: tuple[Telemetry, MagicMock],
        method: str,
        args: tuple[object, ...],
    ) -> None:
        instance, span = telemetry
        sentinel = "0.0.0-release-sentinel"

        # Patched at the source module, not at crewai_core.telemetry: these
        # methods import get_crewai_version inside the call, so a patch on the
        # importing module would never be seen. An arbitrary sentinel also means
        # a hard-coded literal cannot satisfy the assertion.
        with patch("crewai_core.version.get_crewai_version", return_value=sentinel):
            getattr(instance, method)(*args)

        assert _attributes(span).get("crewai_version") == sentinel, (
            f"{method} did not record the value returned by get_crewai_version()"
        )

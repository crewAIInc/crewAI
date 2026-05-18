"""Tests for OpenTelemetry dependency compatibility.

Ensures the opentelemetry version constraints in pyproject.toml are relaxed
enough to avoid conflicts with third-party observability packages (e.g. openlit)
that require newer OpenTelemetry versions.

Regression test for https://github.com/crewAIInc/crewAI/issues/5845
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.version import Version


OTEL_PACKAGES = (
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CREWAI_PYPROJECT = REPO_ROOT / "lib" / "crewai" / "pyproject.toml"
CREWAI_CORE_PYPROJECT = REPO_ROOT / "lib" / "crewai-core" / "pyproject.toml"


def _parse_otel_deps(pyproject_path: Path) -> dict[str, Requirement]:
    """Parse OpenTelemetry dependencies from a pyproject.toml file."""
    with open(pyproject_path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    result: dict[str, Requirement] = {}
    for dep_str in deps:
        req = Requirement(dep_str)
        if req.name in OTEL_PACKAGES:
            result[req.name] = req
    return result


class TestOpenTelemetryVersionConstraints:
    """Verify that OTel dependency constraints are not overly restrictive."""

    @pytest.mark.parametrize("pyproject_path", [CREWAI_PYPROJECT, CREWAI_CORE_PYPROJECT])
    def test_otel_packages_present(self, pyproject_path: Path) -> None:
        """All three OpenTelemetry packages must be declared as dependencies."""
        deps = _parse_otel_deps(pyproject_path)
        for pkg in OTEL_PACKAGES:
            assert pkg in deps, (
                f"{pkg} missing from {pyproject_path.relative_to(REPO_ROOT)}"
            )

    @pytest.mark.parametrize("pyproject_path", [CREWAI_PYPROJECT, CREWAI_CORE_PYPROJECT])
    def test_otel_upper_bound_allows_newer_versions(self, pyproject_path: Path) -> None:
        """OTel constraints must allow versions >= 1.38.0 (required by openlit).

        Regression test for https://github.com/crewAIInc/crewAI/issues/5845
        """
        deps = _parse_otel_deps(pyproject_path)
        for pkg in OTEL_PACKAGES:
            req = deps[pkg]
            # Version 1.38.0 must be within the allowed range
            assert req.specifier.contains(Version("1.38.0")), (
                f"{pkg} in {pyproject_path.relative_to(REPO_ROOT)} does not allow "
                f"version 1.38.0 (specifier: {req.specifier}). This causes dependency "
                f"conflicts with openlit and similar observability packages."
            )

    @pytest.mark.parametrize("pyproject_path", [CREWAI_PYPROJECT, CREWAI_CORE_PYPROJECT])
    def test_otel_lower_bound_is_at_least_1_34(self, pyproject_path: Path) -> None:
        """OTel constraints must have a lower bound >= 1.34.0 for API stability."""
        deps = _parse_otel_deps(pyproject_path)
        for pkg in OTEL_PACKAGES:
            req = deps[pkg]
            assert req.specifier.contains(Version("1.34.0")), (
                f"{pkg} lower bound excludes 1.34.0 (specifier: {req.specifier})"
            )

    @pytest.mark.parametrize("pyproject_path", [CREWAI_PYPROJECT, CREWAI_CORE_PYPROJECT])
    def test_otel_upper_bound_below_2(self, pyproject_path: Path) -> None:
        """OTel constraints must cap at < 2.0.0 to guard against breaking changes."""
        deps = _parse_otel_deps(pyproject_path)
        for pkg in OTEL_PACKAGES:
            req = deps[pkg]
            assert not req.specifier.contains(Version("2.0.0")), (
                f"{pkg} allows version 2.0.0 (specifier: {req.specifier}). "
                f"The upper bound should be < 2.0.0."
            )


class TestOpenTelemetryImports:
    """Verify that the OpenTelemetry APIs used by CrewAI are importable."""

    def test_trace_module_importable(self) -> None:
        mod = importlib.import_module("opentelemetry.trace")
        assert hasattr(mod, "get_tracer")
        assert hasattr(mod, "set_tracer_provider")

    def test_sdk_tracer_provider_importable(self) -> None:
        mod = importlib.import_module("opentelemetry.sdk.trace")
        assert hasattr(mod, "TracerProvider")

    def test_otlp_exporter_importable(self) -> None:
        mod = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
        assert hasattr(mod, "OTLPSpanExporter")

    def test_batch_span_processor_importable(self) -> None:
        mod = importlib.import_module("opentelemetry.sdk.trace.export")
        assert hasattr(mod, "BatchSpanProcessor")
        assert hasattr(mod, "SpanExportResult")

    def test_resource_importable(self) -> None:
        mod = importlib.import_module("opentelemetry.sdk.resources")
        assert hasattr(mod, "SERVICE_NAME")
        assert hasattr(mod, "Resource")

    def test_span_status_importable(self) -> None:
        mod = importlib.import_module("opentelemetry.trace")
        assert hasattr(mod, "Span")
        assert hasattr(mod, "Status")
        assert hasattr(mod, "StatusCode")


class TestCrewaiTelemetryModuleIntegrity:
    """Verify that crewai's telemetry modules import without errors."""

    def test_crewai_core_telemetry_importable(self) -> None:
        mod = importlib.import_module("crewai_core.telemetry")
        assert hasattr(mod, "Telemetry")

    def test_crewai_telemetry_importable(self) -> None:
        mod = importlib.import_module("crewai.telemetry.telemetry")
        assert hasattr(mod, "Telemetry")
        assert hasattr(mod, "SafeOTLPSpanExporter")

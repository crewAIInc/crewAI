"""Shared fixtures for the telemetry suite."""

from crewai_core.telemetry import common_span_attributes
import pytest


@pytest.fixture(autouse=True)
def _reset_common_attributes_cache():
    """Clear the process-wide common-attribute cache around every test.

    ``common_span_attributes`` is cached for the life of the process, which is
    correct in production - a process cannot change which assistant, runtime or
    project it belongs to. In tests it would outlive the environment each test
    sets up, so a value computed under one test's markers would leak into the
    next and make results depend on ordering.
    """
    common_span_attributes.cache_clear()
    yield
    common_span_attributes.cache_clear()

"""Public supported-content-types helper must include structured text formats."""

from crewai_files import get_supported_content_types
from crewai_files.formatting.api import _get_supported_types
from crewai_files.processing.constraints import GEMINI_CONSTRAINTS


def test_helper_matches_formatter_supported_types():
    assert get_supported_content_types("gemini") == _get_supported_types(GEMINI_CONSTRAINTS)


def test_structured_text_formats_included():
    types = get_supported_content_types("gemini")
    assert {"application/json", "application/xml", "application/x-yaml"} <= set(types)

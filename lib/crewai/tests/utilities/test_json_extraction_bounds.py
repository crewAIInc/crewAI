"""Bounds and fragment rejection for _extract_first_json_object
(Cursor Bugbot findings on #9124)."""

import time

from crewai.utilities.converter import _extract_first_json_object


def test_first_object_extracted_with_prefix_junk() -> None:
    assert _extract_first_json_object('Note: {"b": 2}') == {"b": 2}


def test_truncated_outer_rejects_nested_fragment() -> None:
    """'{"a": {"x": 1}' - the inner object parses but is a fragment of the
    truncated outer object; it must not be returned as the result."""
    assert _extract_first_json_object('{"a": {"x": 1}') is None


def test_complete_nested_object_still_parses() -> None:
    assert _extract_first_json_object('{"a": {"x": 1}}') == {"a": {"x": 1}}


def test_pathological_repeating_pattern_is_bounded() -> None:
    """A repeating unclosed pattern made the scan quadratic (~seconds at a
    few hundred KB); the attempt cap bounds it."""
    result = '{"a": "{' * 30000  # ~360KB
    begin = time.monotonic()
    assert _extract_first_json_object(result) is None
    assert time.monotonic() - begin < 5.0

"""A valid JSON object must not be lost because of what follows it."""

from pydantic import BaseModel

from crewai.utilities.converter import handle_partial_json


class Person(BaseModel):
    name: str


def _handle(text: str):
    return handle_partial_json(result=text, model=Person, is_json_output=True, agent=None)


def test_valid_object_followed_by_prose_containing_braces():
    """The greedy pattern used to swallow the trailing note and fail to parse."""
    result = _handle('{"name": "Ada"}\n\nNote: the schema is {"type": "object"}')

    assert result == {"name": "Ada"}


def test_nested_object_is_still_extracted_whole():
    """A non-greedy pattern would cut inside the nested object; raw_decode does not."""
    result = _handle('{"name": "Ada", "extra": {"nested": true}}')

    assert result == {"name": "Ada"}


def test_trailing_closing_brace_is_ignored():
    result = _handle('{"name": "Ada"}}')

    assert result == {"name": "Ada"}

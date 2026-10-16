from typing import Any

from crewai.utilities.pydantic_schema_utils import (
    create_model_from_schema,
    resolve_refs,
    sanitize_tool_params_for_openai_strict,
)
from pydantic import ValidationError
import pytest


@pytest.fixture(params=["Item", "a/b", "a~b", "a~1b", "a/b~c"])
def referenced_schema(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Build a schema with a correctly escaped local definition reference."""
    name = request.param
    token = name.replace("~", "~0").replace("/", "~1")
    return {
        "type": "object",
        "properties": {"item": {"$ref": f"#/$defs/{token}"}},
        "required": ["item"],
        "$defs": {
            name: {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            }
        },
    }


def test_resolve_refs_decodes_definition_tokens(
    referenced_schema: dict[str, Any],
) -> None:
    """Resolve escaped names to their definitions without changing the input."""
    resolved = resolve_refs(referenced_schema)

    assert resolved["properties"]["item"]["properties"]["value"] == {"type": "integer"}
    assert "$ref" in referenced_schema["properties"]["item"]


def test_strict_tool_schema_decodes_definition_tokens(
    referenced_schema: dict[str, Any],
) -> None:
    """Accept escaped definition names in strict OpenAI tool schemas."""
    sanitized = sanitize_tool_params_for_openai_strict(referenced_schema)

    assert sanitized["properties"]["item"]["properties"]["value"]["type"] == ("integer")
    assert "$defs" not in sanitized


def test_recursive_model_decodes_definition_tokens(
    referenced_schema: dict[str, Any],
) -> None:
    """Decode references both at the root and within recursive models."""
    ref = referenced_schema["properties"]["item"]["$ref"]
    definition = next(iter(referenced_schema["$defs"].values()))
    definition["properties"]["children"] = {
        "type": "array",
        "items": {"$ref": ref},
    }
    schema = {"$ref": ref, "$defs": referenced_schema["$defs"]}
    model = create_model_from_schema(schema)

    value = model.model_validate({"value": 1, "children": [{"value": 2}]})
    assert value.model_dump(exclude_none=True) == {
        "value": 1,
        "children": [{"value": 2}],
    }
    with pytest.raises(ValidationError):
        model.model_validate({"value": 1, "children": [{"value": "invalid"}]})

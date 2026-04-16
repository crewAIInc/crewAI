"""Tests for pydantic_schema_utils module.

Covers:
- create_model_from_schema: type mapping, required/optional, enums, formats,
  nested objects, arrays, unions, allOf, $ref, model_name, enrich_descriptions
- Schema transformation helpers: resolve_refs, force_additional_properties_false,
  strip_unsupported_formats, ensure_type_in_schemas, convert_oneof_to_anyof,
  ensure_all_properties_required, strip_null_from_types, build_rich_field_description
- End-to-end MCP tool schema conversion
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from crewai.utilities.pydantic_schema_utils import (
    build_rich_field_description,
    convert_oneof_to_anyof,
    create_model_from_schema,
    ensure_all_properties_required,
    ensure_type_in_schemas,
    force_additional_properties_false,
    resolve_refs,
    strip_null_from_types,
    strip_unsupported_formats,
)


class TestSimpleTypes:
    def test_string_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(name="Alice")
        assert obj.name == "Alice"

    def test_integer_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(count=42)
        assert obj.count == 42

    def test_number_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(score=3.14)
        assert obj.score == pytest.approx(3.14)

    def test_boolean_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        }
        Model = create_model_from_schema(schema)
        assert Model(active=True).active is True

    def test_null_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "null"}},
            "required": ["value"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(value=None)
        assert obj.value is None


class TestRequiredOptional:
    def test_required_field_has_no_default(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        Model = create_model_from_schema(schema)
        with pytest.raises(Exception):
            Model()

    def test_optional_field_defaults_to_none(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": [],
        }
        Model = create_model_from_schema(schema)
        obj = Model()
        assert obj.name is None

    def test_mixed_required_optional(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "label": {"type": "string"},
            },
            "required": ["id"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(id=1)
        assert obj.id == 1
        assert obj.label is None


class TestEnumLiteral:
    def test_string_enum(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
            },
            "required": ["color"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(color="red")
        assert obj.color == "red"

    def test_string_enum_rejects_invalid(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
            },
            "required": ["color"],
        }
        Model = create_model_from_schema(schema)
        with pytest.raises(Exception):
            Model(color="yellow")

    def test_const_value(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "kind": {"const": "fixed"},
            },
            "required": ["kind"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(kind="fixed")
        assert obj.kind == "fixed"


class TestFormatMapping:
    def test_date_format(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "birthday": {"type": "string", "format": "date"},
            },
            "required": ["birthday"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(birthday=datetime.date(2000, 1, 15))
        assert obj.birthday == datetime.date(2000, 1, 15)

    def test_datetime_format(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"},
            },
            "required": ["created_at"],
        }
        Model = create_model_from_schema(schema)
        dt = datetime.datetime(2025, 6, 1, 12, 0, 0)
        obj = Model(created_at=dt)
        assert obj.created_at == dt

    def test_time_format(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "alarm": {"type": "string", "format": "time"},
            },
            "required": ["alarm"],
        }
        Model = create_model_from_schema(schema)
        t = datetime.time(8, 30)
        obj = Model(alarm=t)
        assert obj.alarm == t


class TestNestedObjects:
    def test_nested_object_creates_model(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                },
            },
            "required": ["address"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(address={"street": "123 Main", "city": "Springfield"})
        assert obj.address.street == "123 Main"
        assert obj.address.city == "Springfield"

    def test_object_without_properties_returns_dict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
            "required": ["metadata"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(metadata={"key": "value"})
        assert obj.metadata == {"key": "value"}


class TestTypedArrays:
    def test_array_of_strings(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(tags=["a", "b", "c"])
        assert obj.tags == ["a", "b", "c"]

    def test_array_of_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                },
            },
            "required": ["items"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(items=[{"id": 1}, {"id": 2}])
        assert len(obj.items) == 2
        assert obj.items[0].id == 1

    def test_untyped_array(self) -> None:
        schema = {
            "type": "object",
            "properties": {"data": {"type": "array"}},
            "required": ["data"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(data=[1, "two", 3.0])
        assert obj.data == [1, "two", 3.0]


class TestUnionTypes:
    def test_anyof_string_or_integer(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
            },
            "required": ["value"],
        }
        Model = create_model_from_schema(schema)
        assert Model(value="hello").value == "hello"
        assert Model(value=42).value == 42

    def test_oneof(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [{"type": "string"}, {"type": "number"}],
                },
            },
            "required": ["value"],
        }
        Model = create_model_from_schema(schema)
        assert Model(value="hello").value == "hello"
        assert Model(value=3.14).value == pytest.approx(3.14)


class TestAllOfMerging:
    def test_allof_merges_properties(self) -> None:
        schema = {
            "type": "object",
            "allOf": [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"],
                },
            ],
        }
        Model = create_model_from_schema(schema)
        obj = Model(name="Alice", age=30)
        assert obj.name == "Alice"
        assert obj.age == 30

    def test_single_allof(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                            "required": ["id"],
                        }
                    ]
                }
            },
            "required": ["item"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(item={"id": 1})
        assert obj.item.id == 1


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


class TestRefResolution:
    def test_ref_in_property(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "required": ["item"],
            "$defs": {
                "Item": {
                    "type": "object",
                    "title": "Item",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
        Model = create_model_from_schema(schema)
        obj = Model(item={"name": "Widget"})
        assert obj.item.name == "Widget"


# ---------------------------------------------------------------------------
# model_name parameter
# ---------------------------------------------------------------------------


class TestModelName:
    def test_model_name_override(self) -> None:
        schema = {
            "type": "object",
            "title": "OriginalName",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        Model = create_model_from_schema(schema, model_name="CustomSchema")
        assert Model.__name__ == "CustomSchema"

    def test_model_name_fallback_to_title(self) -> None:
        schema = {
            "type": "object",
            "title": "FromTitle",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        Model = create_model_from_schema(schema)
        assert Model.__name__ == "FromTitle"

    def test_model_name_fallback_to_dynamic(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        Model = create_model_from_schema(schema)
        assert Model.__name__ == "DynamicModel"


# ---------------------------------------------------------------------------
# enrich_descriptions
# ---------------------------------------------------------------------------


class TestEnrichDescriptions:
    def test_enriched_description_includes_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "The score value",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["score"],
        }
        Model = create_model_from_schema(schema, enrich_descriptions=True)
        field_info = Model.model_fields["score"]
        assert "Minimum: 0" in field_info.description
        assert "Maximum: 100" in field_info.description
        assert "The score value" in field_info.description

    def test_default_does_not_enrich(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "The score value",
                    "minimum": 0,
                },
            },
            "required": ["score"],
        }
        Model = create_model_from_schema(schema, enrich_descriptions=False)
        field_info = Model.model_fields["score"]
        assert field_info.description == "The score value"

    def test_enriched_description_propagates_to_nested(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "integer",
                            "description": "Level",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["level"],
                },
            },
            "required": ["config"],
        }
        Model = create_model_from_schema(schema, enrich_descriptions=True)
        nested_model = Model.model_fields["config"].annotation
        nested_field = nested_model.model_fields["level"]
        assert "Minimum: 1" in nested_field.description
        assert "Maximum: 10" in nested_field.description


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_properties(self) -> None:
        schema = {"type": "object", "properties": {}, "required": []}
        Model = create_model_from_schema(schema)
        obj = Model()
        assert obj is not None

    def test_no_properties_key(self) -> None:
        schema = {"type": "object"}
        Model = create_model_from_schema(schema)
        obj = Model()
        assert obj is not None

    def test_unknown_type_raises(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "weird": {"type": "hyperspace"},
            },
            "required": ["weird"],
        }
        with pytest.raises(ValueError, match="Unsupported JSON schema type"):
            create_model_from_schema(schema)


# ---------------------------------------------------------------------------
# build_rich_field_description
# ---------------------------------------------------------------------------


class TestBuildRichFieldDescription:
    def test_description_only(self) -> None:
        assert build_rich_field_description({"description": "A name"}) == "A name"

    def test_empty_schema(self) -> None:
        assert build_rich_field_description({}) == ""

    def test_format(self) -> None:
        desc = build_rich_field_description({"format": "date-time"})
        assert "Format: date-time" in desc

    def test_enum(self) -> None:
        desc = build_rich_field_description({"enum": ["a", "b"]})
        assert "Allowed values:" in desc
        assert "'a'" in desc
        assert "'b'" in desc

    def test_pattern(self) -> None:
        desc = build_rich_field_description({"pattern": "^[a-z]+$"})
        assert "Pattern: ^[a-z]+$" in desc

    def test_min_max(self) -> None:
        desc = build_rich_field_description({"minimum": 0, "maximum": 100})
        assert "Minimum: 0" in desc
        assert "Maximum: 100" in desc

    def test_min_max_length(self) -> None:
        desc = build_rich_field_description({"minLength": 1, "maxLength": 255})
        assert "Min length: 1" in desc
        assert "Max length: 255" in desc

    def test_examples(self) -> None:
        desc = build_rich_field_description({"examples": ["foo", "bar", "baz", "extra"]})
        assert "Examples:" in desc
        assert "'foo'" in desc
        assert "'baz'" in desc
        # Only first 3 shown
        assert "'extra'" not in desc

    def test_combined_constraints(self) -> None:
        desc = build_rich_field_description({
            "description": "A score",
            "minimum": 0,
            "maximum": 10,
            "format": "int32",
        })
        assert desc.startswith("A score")
        assert "Minimum: 0" in desc
        assert "Maximum: 10" in desc
        assert "Format: int32" in desc


# ---------------------------------------------------------------------------
# Schema transformation functions
# ---------------------------------------------------------------------------


class TestResolveRefs:
    def test_basic_ref_resolution(self) -> None:
        schema = {
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}},
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}},
            },
        }
        resolved = resolve_refs(schema)
        assert "$ref" not in resolved["properties"]["item"]
        assert resolved["properties"]["item"]["type"] == "object"

    def test_nested_ref_resolution(self) -> None:
        schema = {
            "type": "object",
            "properties": {"wrapper": {"$ref": "#/$defs/Wrapper"}},
            "$defs": {
                "Wrapper": {
                    "type": "object",
                    "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                },
                "Inner": {"type": "string"},
            },
        }
        resolved = resolve_refs(schema)
        wrapper = resolved["properties"]["wrapper"]
        assert wrapper["properties"]["inner"]["type"] == "string"

    def test_missing_ref_raises(self) -> None:
        schema = {
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
            "$defs": {},
        }
        with pytest.raises(KeyError, match="Missing"):
            resolve_refs(schema)

    def test_no_refs_unchanged(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        resolved = resolve_refs(schema)
        assert resolved == schema


class TestForceAdditionalPropertiesFalse:
    def test_adds_to_object(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        result = force_additional_properties_false(deepcopy(schema))
        assert result["additionalProperties"] is False

    def test_adds_empty_properties_and_required(self) -> None:
        schema = {"type": "object"}
        result = force_additional_properties_false(deepcopy(schema))
        assert result["properties"] == {}
        assert result["required"] == []

    def test_recursive_nested_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
        }
        result = force_additional_properties_false(deepcopy(schema))
        assert result["additionalProperties"] is False
        assert result["properties"]["child"]["additionalProperties"] is False

    def test_does_not_affect_non_objects(self) -> None:
        schema = {"type": "string"}
        result = force_additional_properties_false(deepcopy(schema))
        assert "additionalProperties" not in result


class TestStripUnsupportedFormats:
    def test_removes_email_format(self) -> None:
        schema = {"type": "string", "format": "email"}
        result = strip_unsupported_formats(deepcopy(schema))
        assert "format" not in result

    def test_keeps_date_time(self) -> None:
        schema = {"type": "string", "format": "date-time"}
        result = strip_unsupported_formats(deepcopy(schema))
        assert result["format"] == "date-time"

    def test_keeps_date(self) -> None:
        schema = {"type": "string", "format": "date"}
        result = strip_unsupported_formats(deepcopy(schema))
        assert result["format"] == "date"

    def test_removes_uri_format(self) -> None:
        schema = {"type": "string", "format": "uri"}
        result = strip_unsupported_formats(deepcopy(schema))
        assert "format" not in result

    def test_recursive(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "created": {"type": "string", "format": "date-time"},
            },
        }
        result = strip_unsupported_formats(deepcopy(schema))
        assert "format" not in result["properties"]["email"]
        assert result["properties"]["created"]["format"] == "date-time"


class TestEnsureTypeInSchemas:
    def test_empty_schema_in_anyof_gets_type(self) -> None:
        schema = {"anyOf": [{}, {"type": "string"}]}
        result = ensure_type_in_schemas(deepcopy(schema))
        assert result["anyOf"][0] == {"type": "object"}

    def test_empty_schema_in_oneof_gets_type(self) -> None:
        schema = {"oneOf": [{}, {"type": "integer"}]}
        result = ensure_type_in_schemas(deepcopy(schema))
        assert result["oneOf"][0] == {"type": "object"}

    def test_non_empty_unchanged(self) -> None:
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = ensure_type_in_schemas(deepcopy(schema))
        assert result == schema


class TestConvertOneofToAnyof:
    def test_converts_top_level(self) -> None:
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        result = convert_oneof_to_anyof(deepcopy(schema))
        assert "oneOf" not in result
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_converts_nested(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {"oneOf": [{"type": "string"}, {"type": "number"}]},
            },
        }
        result = convert_oneof_to_anyof(deepcopy(schema))
        assert "anyOf" in result["properties"]["value"]
        assert "oneOf" not in result["properties"]["value"]


class TestEnsureAllPropertiesRequired:
    def test_makes_all_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        result = ensure_all_properties_required(deepcopy(schema))
        assert set(result["required"]) == {"a", "b"}

    def test_recursive(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": [],
                },
            },
        }
        result = ensure_all_properties_required(deepcopy(schema))
        assert set(result["properties"]["child"]["required"]) == {"x", "y"}


class TestStripNullFromTypes:
    def test_strips_null_from_anyof(self) -> None:
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
        }
        result = strip_null_from_types(deepcopy(schema))
        assert "anyOf" not in result
        assert result["type"] == "string"

    def test_strips_null_from_type_array(self) -> None:
        schema = {"type": ["string", "null"]}
        result = strip_null_from_types(deepcopy(schema))
        assert result["type"] == "string"

    def test_multiple_non_null_in_anyof(self) -> None:
        schema = {
            "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}],
        }
        result = strip_null_from_types(deepcopy(schema))
        assert len(result["anyOf"]) == 2

    def test_no_null_unchanged(self) -> None:
        schema = {"type": "string"}
        result = strip_null_from_types(deepcopy(schema))
        assert result == schema


class TestEndToEndMCPSchema:
    """Realistic MCP tool schema exercising multiple features simultaneously."""

    MCP_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "minLength": 1,
                "maxLength": 500,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results",
                "minimum": 1,
                "maximum": 100,
            },
            "format": {
                "type": "string",
                "enum": ["json", "csv", "xml"],
                "description": "Output format",
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "format": "date"},
                    "date_to": {"type": "string", "format": "date"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["date_from"],
            },
            "sort_order": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        },
        "required": ["query", "format", "filters"],
    }

    def test_model_creation(self) -> None:
        Model = create_model_from_schema(self.MCP_SCHEMA)
        assert Model is not None
        assert issubclass(Model, BaseModel)

    def test_valid_input_accepted(self) -> None:
        Model = create_model_from_schema(self.MCP_SCHEMA)
        obj = Model(
            query="test search",
            format="json",
            filters={"date_from": "2025-01-01"},
        )
        assert obj.query == "test search"
        assert obj.format == "json"

    def test_invalid_enum_rejected(self) -> None:
        Model = create_model_from_schema(self.MCP_SCHEMA)
        with pytest.raises(Exception):
            Model(
                query="test",
                format="yaml",
                filters={"date_from": "2025-01-01"},
            )

    def test_model_name_for_mcp_tool(self) -> None:
        Model = create_model_from_schema(
            self.MCP_SCHEMA, model_name="search_toolSchema"
        )
        assert Model.__name__ == "search_toolSchema"

    def test_enriched_descriptions_for_mcp(self) -> None:
        Model = create_model_from_schema(
            self.MCP_SCHEMA, enrich_descriptions=True
        )
        query_field = Model.model_fields["query"]
        assert "Min length: 1" in query_field.description
        assert "Max length: 500" in query_field.description

        max_results_field = Model.model_fields["max_results"]
        assert "Minimum: 1" in max_results_field.description
        assert "Maximum: 100" in max_results_field.description

        format_field = Model.model_fields["format"]
        assert "Allowed values:" in format_field.description

    def test_optional_fields_accept_none(self) -> None:
        Model = create_model_from_schema(self.MCP_SCHEMA)
        obj = Model(
            query="test",
            format="csv",
            filters={"date_from": "2025-01-01"},
            max_results=None,
            sort_order=None,
        )
        assert obj.max_results is None
        assert obj.sort_order is None

    def test_nested_filters_validated(self) -> None:
        Model = create_model_from_schema(self.MCP_SCHEMA)
        obj = Model(
            query="test",
            format="xml",
            filters={
                "date_from": "2025-01-01",
                "date_to": "2025-12-31",
                "categories": ["news", "tech"],
            },
        )
        assert obj.filters.date_from == datetime.date(2025, 1, 1)
        assert obj.filters.categories == ["news", "tech"]

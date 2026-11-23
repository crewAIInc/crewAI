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

from copy import deepcopy
import datetime
from pathlib import Path
import time
from typing import Any

from pydantic import AnyUrl, BaseModel, SecretStr, ValidationError
import pytest

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

    def test_field_with_explicit_non_null_default_stays_non_nullable(self) -> None:
        """A field omitted from ``required`` but carrying an explicit default
        is already usable when omitted, so it must not be made nullable.

        This is the ``model_json_schema() -> create_model_from_schema()``
        round-trip used to persist tool schemas: ``limit: int = 10`` produces
        ``{"type": "integer", "default": 10}`` and must not come back as
        ``Optional[int]`` that silently accepts ``None``.
        """
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "include_snippets": {"type": "boolean", "default": True},
                "note": {"type": "string", "default": ""},
            },
            "required": ["query"],
        }
        Model = create_model_from_schema(schema)

        assert Model.model_fields["limit"].annotation is int
        assert Model.model_fields["include_snippets"].annotation is bool
        assert Model.model_fields["note"].annotation is str

        obj = Model(query="x")
        assert obj.limit == 10
        assert obj.include_snippets is True
        assert obj.note == ""

        for field in ("limit", "include_snippets", "note"):
            with pytest.raises(Exception):
                Model(query="x", **{field: None})

    def test_field_with_explicit_null_default_remains_nullable(self) -> None:
        """An explicit ``default: null`` must keep accepting ``None``."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": None},
            },
            "required": ["query"],
        }
        Model = create_model_from_schema(schema)
        assert Model(query="x").limit is None
        assert Model(query="x", limit=None).limit is None

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

    def test_type_array_nullable_string_with_format(self) -> None:
        """type: ["string", "null"] -- the .NET/System.Text.Json-style way
        of expressing an optional field, as opposed to Pydantic's own
        anyOf-based form. Seen in real MCP tool schemas from non-Python
        servers (e.g. Equibles' ListCompanyDocuments startDate/endDate
        filters). The format="date-time" here (also straight from that
        real schema) is applied by the existing FORMAT_TYPE_MAP logic once
        the list-form type no longer raises, so the field lands as a real
        datetime rather than str -- that's the pre-existing, correct
        behavior for any date-time-formatted field, not something this fix
        changes."""
        schema = {
            "type": "object",
            "properties": {
                "startDate": {
                    "description": "Optional start date filter in YYYY-MM-DD format",
                    "type": ["string", "null"],
                    "format": "date-time",
                    "default": None,
                },
            },
        }
        Model = create_model_from_schema(schema)
        assert Model(startDate="2026-01-01").startDate == datetime.datetime(
            2026, 1, 1
        )
        assert Model(startDate=None).startDate is None
        assert Model().startDate is None

    def test_type_array_nullable_string_no_format(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "note": {"type": ["string", "null"]},
            },
        }
        Model = create_model_from_schema(schema)
        assert Model(note="hello").note == "hello"
        assert Model(note=None).note is None
        assert Model().note is None

    def test_type_array_multiple_non_null(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "integer", "null"]},
            },
        }
        Model = create_model_from_schema(schema)
        assert Model(value="hello").value == "hello"
        assert Model(value=42).value == 42
        assert Model(value=None).value is None

    def test_type_array_single_element(self) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": ["string"]}},
            "required": ["value"],
        }
        Model = create_model_from_schema(schema)
        assert Model(value="hello").value == "hello"

    def test_type_array_required_nullable_string_with_format(self) -> None:
        """A required-but-nullable formatted field, e.g. `{"type":
        ["string", "null"], "format": "date-time"}` inside a "required"
        list -- a valid JSON Schema shape meaning the key must be present
        but its value may be null. Before this fix, the FORMAT_TYPE_MAP
        override in `_json_schema_to_pydantic_field` replaced the whole
        `Union[datetime, None]` with plain `datetime`, so passing `None`
        would fail validation even though the schema explicitly allows it.
        The `not is_required` Optional-rewrap at the end of that function
        doesn't fire for required fields, so this case wasn't masked the
        way the non-required version (test above) was.
        """
        schema = {
            "type": "object",
            "properties": {
                "startDate": {
                    "type": ["string", "null"],
                    "format": "date-time",
                },
            },
            "required": ["startDate"],
        }
        Model = create_model_from_schema(schema)
        assert Model(startDate="2026-01-01").startDate == datetime.datetime(
            2026, 1, 1
        )
        assert Model(startDate=None).startDate is None
        with pytest.raises(Exception):
            Model()

    def test_type_array_multiple_non_null_with_format(self) -> None:
        """A list-form type with more than one non-null member plus a
        recognized format, e.g. `{"type": ["string", "integer", "null"],
        "format": "date-time"}`. Before this fix, the FORMAT_TYPE_MAP
        override collapsed the entire Union down to plain `datetime`,
        silently dropping the `integer` alternative regardless of whether
        the field was required. The fix narrows only the `str` member of
        the union to the formatted type, leaving `integer` and `None`
        alone.
        """
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": ["string", "integer", "null"],
                    "format": "date-time",
                },
            },
        }
        Model = create_model_from_schema(schema)
        assert Model(value="2026-01-01").value == datetime.datetime(2026, 1, 1)
        assert Model(value=42).value == 42
        assert Model(value=None).value is None


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

    def test_allof_preserves_outer_properties_required_and_title(self) -> None:
        schema = {
            "title": "AdminUser",
            "type": "object",
            "allOf": [{"$ref": "#/$defs/BaseUser"}],
            "properties": {
                "role": {"type": "string"},
            },
            "required": ["role"],
            "$defs": {
                "BaseUser": {
                    "title": "BaseUser",
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                    },
                    "required": ["username"],
                }
            },
        }
        Model = create_model_from_schema(schema)
        assert Model.__name__ == "AdminUser"
        assert set(Model.model_fields) == {"username", "role"}
        obj = Model(username="alice", role="admin")
        assert obj.username == "alice"
        assert obj.role == "admin"
        with pytest.raises(ValidationError):
            Model(username="alice")
        with pytest.raises(ValidationError):
            Model(role="admin")

    def test_nested_single_allof_preserves_sibling_properties(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                }
            },
            "properties": {
                "admin": {
                    "type": "object",
                    "allOf": [{"$ref": "#/$defs/Base"}],
                    "properties": {"role": {"type": "string"}},
                    "required": ["role"],
                }
            },
            "required": ["admin"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(admin={"id": 7, "role": "owner"})
        assert obj.admin.id == 7
        assert obj.admin.role == "owner"
        with pytest.raises(ValidationError):
            Model(admin={"id": 7})

    def test_nested_multilevel_allof_inheritance(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                },
                "Admin": {
                    "allOf": [{"$ref": "#/$defs/Base"}],
                    "properties": {"role": {"type": "string"}},
                    "required": ["role"],
                },
            },
            "title": "SuperAdmin",
            "allOf": [{"$ref": "#/$defs/Admin"}],
            "properties": {"level": {"type": "integer"}},
            "required": ["level"],
        }
        Model = create_model_from_schema(schema)
        assert Model.__name__ == "SuperAdmin"
        obj = Model(id=1, role="root", level=99)
        assert (obj.id, obj.role, obj.level) == (1, "root", 99)

    def test_scalar_string_allof_preserves_type_and_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "code": {
                    "title": "Code",
                    "allOf": [
                        {"type": "string", "minLength": 2},
                        {"type": "string", "maxLength": 5, "pattern": "^[a-z]+$"},
                    ],
                }
            },
            "required": ["code"],
        }
        Model = create_model_from_schema(schema)
        assert Model.model_fields["code"].annotation is str
        assert Model(code="abc").code == "abc"
        for invalid in ("a", "abcdef", "123"):
            with pytest.raises(ValidationError):
                Model(code=invalid)

    def test_scalar_integer_allof_tightens_bounds(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "allOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "integer", "minimum": 5, "maximum": 100},
                        {"maximum": 10},
                    ]
                }
            },
            "required": ["score"],
        }
        Model = create_model_from_schema(schema)
        assert Model.model_fields["score"].annotation is int
        assert Model(score=7).score == 7
        for invalid in (4, 11):
            with pytest.raises(ValidationError):
                Model(score=invalid)

    def test_scalar_array_and_enum_allof(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "allOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"minItems": 1},
                    ]
                },
                "status": {
                    "allOf": [
                        {"type": "string", "enum": ["open", "closed", "pending"]},
                        {"enum": ["closed", "pending"]},
                    ]
                },
                "meta": {
                    "allOf": [
                        {"type": "object"},
                        {"description": "Free-form metadata"},
                    ]
                },
            },
            "required": ["tags", "status", "meta"],
        }
        Model = create_model_from_schema(schema)
        assert Model.model_fields["tags"].annotation == list[str]
        assert Model.model_fields["meta"].annotation is dict
        obj = Model(tags=["v1"], status="closed", meta={"k": "v"})
        assert obj.tags == ["v1"]
        assert obj.status == "closed"
        assert obj.meta == {"k": "v"}
        with pytest.raises(ValidationError):
            Model(tags=["v1"], status="open", meta={})

    def test_single_allof_ref_preserves_format_and_outer_description(self) -> None:
        schema = {
            "type": "object",
            "$defs": {
                "Timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Generic timestamp",
                }
            },
            "properties": {
                "created_at": {
                    "allOf": [{"$ref": "#/$defs/Timestamp"}],
                    "description": "Created at",
                }
            },
            "required": ["created_at"],
        }
        Model = create_model_from_schema(schema)
        field_info = Model.model_fields["created_at"]
        assert field_info.annotation is datetime.datetime
        assert field_info.description == "Created at"
        obj = Model(created_at="2026-11-23T12:00:00Z")
        assert isinstance(obj.created_at, datetime.datetime)

    def test_recursive_ref_inside_allof_does_not_recurse_infinitely(self) -> None:
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "title": "Node",
                    "properties": {
                        "label": {"type": "string"},
                        "parent": {
                            "allOf": [{"$ref": "#/$defs/Node"}],
                            "description": "Parent node",
                        },
                    },
                    "required": ["label"],
                }
            },
            "$ref": "#/$defs/Node",
        }
        Model = create_model_from_schema(deepcopy(schema), model_name="Node")
        assert Model.model_fields["parent"].description == "Parent node"
        obj = Model(label="root", parent={"label": "top"})
        assert obj.label == "root"
        assert obj.parent is not None
        assert obj.parent.label == "top"

    def test_description_only_sibling_property_preserves_allof_constraints(self) -> None:
        schema = {
            "type": "object",
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "pattern": "^(ls|pwd)$",
                            "minLength": 2,
                        }
                    },
                    "required": ["cmd"],
                }
            ],
            "properties": {
                "cmd": {"description": "Allowed shell command"},
            },
        }
        Model = create_model_from_schema(schema)
        assert Model.model_fields["cmd"].description == "Allowed shell command"
        assert Model(cmd="ls").cmd == "ls"
        for forbidden in ("rm -rf /", "a", "ls -la"):
            with pytest.raises(ValidationError):
                Model(cmd=forbidden)

    def test_multiple_patterns_in_allof_are_conjunctive(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "pattern": "^[a-z0-9_]+$",
                    "allOf": [
                        {"pattern": "^safe_"},
                        {"pattern": "_v[0-9]+$"},
                    ],
                }
            },
            "required": ["token"],
        }
        Model = create_model_from_schema(schema)
        assert Model(token="safe_item_v2").token == "safe_item_v2"
        for invalid in ("unsafe_item_v2", "safe_item", "safe_BAD_v2"):
            with pytest.raises(ValidationError):
                Model(token=invalid)

    def test_disjoint_enums_conflicting_const_and_incompatible_types_fail_closed(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "disjoint_enum": {
                    "allOf": [{"enum": ["read"]}, {"enum": ["write"]}]
                },
                "conflicting_const": {
                    "allOf": [{"const": 1}, {"const": True}]
                },
                "incompatible_type": {
                    "allOf": [{"type": "string"}, {"type": "integer"}]
                },
                "inverted_bounds": {
                    "allOf": [{"type": "string", "minLength": 5}, {"maxLength": 2}]
                },
            },
            "required": [
                "disjoint_enum",
                "conflicting_const",
                "incompatible_type",
                "inverted_bounds",
            ],
        }
        Model = create_model_from_schema(schema)
        for vals in [
            {
                "disjoint_enum": "read",
                "conflicting_const": 1,
                "incompatible_type": "x",
                "inverted_bounds": "abc",
            },
            {
                "disjoint_enum": "write",
                "conflicting_const": True,
                "incompatible_type": 1,
                "inverted_bounds": None,
            },
        ]:
            with pytest.raises(ValidationError):
                Model(**vals)

        top_conflict = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "allOf": [{"type": "string"}],
        }
        TopConflictModel = create_model_from_schema(top_conflict)
        with pytest.raises(ValidationError):
            TopConflictModel(a="x")

    def test_enum_and_const_enforce_merged_string_pattern_and_length(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["cmd_ok", "bad_123"],
                    "allOf": [{"pattern": "^[a-z_]+$", "minLength": 5}],
                },
                "pinned": {
                    "allOf": [
                        {"const": "safe_cmd"},
                        {"type": "string", "pattern": "^safe_[a-z]+$"},
                    ]
                },
                "unsat_const": {
                    "allOf": [{"const": "a"}, {"type": "string", "minLength": 3}]
                },
            },
            "required": ["action", "pinned"],
        }
        Model = create_model_from_schema(schema)
        assert Model(action="cmd_ok", pinned="safe_cmd").action == "cmd_ok"
        with pytest.raises(ValidationError):
            Model(action="bad_123", pinned="safe_cmd")
        with pytest.raises(ValidationError):
            Model(action="cmd_ok", pinned="other")
        with pytest.raises(ValidationError):
            Model(action="cmd_ok", pinned="safe_cmd", unsat_const="a")

    def test_anyof_branches_intersect_across_allof(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {
                            "anyOf": [
                                {"type": "string", "pattern": "^foo"},
                                {"type": "integer", "minimum": 10},
                            ]
                        },
                        {
                            "anyOf": [
                                {"type": "string", "pattern": "bar$"},
                                {"type": "string", "pattern": "^baz"},
                            ]
                        },
                    ]
                }
            },
            "required": ["value"],
        }
        Model = create_model_from_schema(schema)
        assert Model(value="foobar").value == "foobar"
        for forbidden in (15, "foo", "bar", "baz"):
            with pytest.raises(ValidationError):
                Model(value=forbidden)

    def test_deep_diamond_allof_items_dag_completes_linearly(self) -> None:
        defs: dict[str, Any] = {
            "Leaf": {"type": "string", "minLength": 1},
        }
        depth = 30
        prev = "Leaf"
        for i in range(depth):
            left = f"L{i}_A"
            right = f"L{i}_B"
            defs[left] = {"type": "array", "items": {"$ref": f"#/$defs/{prev}"}}
            defs[right] = {"type": "array", "items": {"$ref": f"#/$defs/{prev}"}}
            node = f"N{i}"
            defs[node] = {"allOf": [{"$ref": f"#/$defs/{left}"}, {"$ref": f"#/$defs/{right}"}]}
            prev = node
        schema = {
            "$defs": defs,
            "type": "object",
            "properties": {"payload": {"$ref": f"#/$defs/{prev}"}},
            "required": ["payload"],
        }
        start = time.monotonic()
        Model = create_model_from_schema(schema)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0
        nested: Any = "ok"
        for _ in range(depth):
            nested = [nested]
        assert Model(payload=nested).payload == nested

    def test_all_of_array_items_enforces_scalar_allowlist(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "cmds": {
                    "type": "array",
                    "items": {
                        "allOf": [
                            {"type": "string", "pattern": "^ls$", "minLength": 2},
                            {"description": "Allowed command"},
                        ]
                    },
                },
                "single_cmds": {
                    "type": "array",
                    "items": {
                        "allOf": [
                            {"type": "string", "pattern": "^safe_[0-9]+$"},
                        ]
                    },
                },
                "even_ints": {
                    "type": "array",
                    "items": {
                        "allOf": [
                            {"type": "integer", "minimum": 2, "maximum": 10},
                            {"multipleOf": 2},
                        ]
                    },
                },
            },
            "required": ["cmds", "single_cmds", "even_ints"],
        }
        Model = create_model_from_schema(schema)
        obj = Model(cmds=["ls"], single_cmds=["safe_1"], even_ints=[2, 10])
        assert obj.cmds == ["ls"]
        assert obj.single_cmds == ["safe_1"]
        assert obj.even_ints == [2, 10]
        for bad in (
            {"cmds": ["rm -rf /"], "single_cmds": ["safe_1"], "even_ints": [2]},
            {"cmds": ["x"], "single_cmds": ["safe_1"], "even_ints": [2]},
            {"cmds": ["ls"], "single_cmds": ["rm -rf /"], "even_ints": [2]},
            {"cmds": ["ls"], "single_cmds": ["safe_1"], "even_ints": [3]},
            {"cmds": ["ls"], "single_cmds": ["safe_1"], "even_ints": [12]},
        ):
            with pytest.raises(ValidationError):
                Model(**bad)

    def test_all_of_intersecting_single_pattern_any_of_arms(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "cmd": {
                    "allOf": [
                        {"type": "string"},
                        {
                            "anyOf": [
                                {"pattern": "^ls$"},
                                {"pattern": "^pwd$"},
                            ]
                        },
                    ]
                }
            },
            "required": ["cmd"],
        }
        Model = create_model_from_schema(schema)
        assert Model(cmd="ls").cmd == "ls"
        assert Model(cmd="pwd").cmd == "pwd"
        with pytest.raises(ValidationError):
            Model(cmd="rm")

    def test_all_of_redos_pattern_uses_linear_engine(self) -> None:
        near_miss = "a" * 40 + "!"
        const_schema = {
            "type": "object",
            "properties": {
                "tok": {
                    "allOf": [
                        {"const": near_miss},
                        {"type": "string", "pattern": "^(a+)+$"},
                    ]
                }
            },
            "required": ["tok"],
        }
        t0 = time.monotonic()
        ConstModel = create_model_from_schema(const_schema)
        assert time.monotonic() - t0 < 1.0
        with pytest.raises(ValidationError):
            ConstModel(tok=near_miss)

        multi_schema = {
            "type": "object",
            "properties": {
                "tok": {
                    "allOf": [
                        {"type": "string", "pattern": "^(a+)+$"},
                        {"pattern": "^(a*)*b$"},
                    ]
                }
            },
            "required": ["tok"],
        }
        MultiModel = create_model_from_schema(multi_schema)
        t1 = time.monotonic()
        with pytest.raises(ValidationError):
            MultiModel(tok=near_miss)
        assert time.monotonic() - t1 < 1.0

        lookaround_schema = {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "enum": ["ls", "cat"]},
                "tok": {
                    "allOf": [
                        {"type": "string", "pattern": "^(?!bad).*$"},
                        {"pattern": "^[a-z]+$"},
                    ]
                },
            },
            "required": ["cmd"],
        }
        LookaroundModel = create_model_from_schema(lookaround_schema)
        assert LookaroundModel(cmd="ls").cmd == "ls"
        with pytest.raises(ValidationError):
            LookaroundModel(cmd="rm")
        with pytest.raises(ValidationError, match="Unsupported or invalid schema pattern"):
            LookaroundModel(cmd="ls", tok="good")

        oversized_schema = {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "enum": ["ls", "cat"]},
                "tok": {
                    "allOf": [
                        {"type": "string", "pattern": "a" * 1025},
                        {"minLength": 1},
                    ]
                },
            },
            "required": ["cmd"],
        }
        OversizedModel = create_model_from_schema(oversized_schema)
        assert OversizedModel(cmd="cat").cmd == "cat"
        with pytest.raises(ValidationError):
            OversizedModel(cmd="rm")
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            OversizedModel(cmd="cat", tok="aaa")

    def test_union_boolean_false_arm_fails_closed(self) -> None:
        any_of_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "anyOf": [False, {"type": "string", "enum": ["read"]}],
                },
            },
            "required": ["action"],
        }
        AnyOfModel = create_model_from_schema(any_of_schema)
        assert AnyOfModel(action="read").action == "read"
        for bad in ("write", 123):
            with pytest.raises(ValidationError):
                AnyOfModel(action=bad)

        one_of_schema = {
            "type": "object",
            "properties": {
                "tok": {
                    "oneOf": [False, {"type": "string", "pattern": "^ok$"}],
                },
            },
            "required": ["tok"],
        }
        OneOfModel = create_model_from_schema(one_of_schema)
        assert OneOfModel(tok="ok").tok == "ok"
        with pytest.raises(ValidationError):
            OneOfModel(tok="bad")

        all_false_schema = {
            "type": "object",
            "properties": {"blocked": {"anyOf": [False]}},
            "required": ["blocked"],
        }
        BlockedModel = create_model_from_schema(all_false_schema)
        with pytest.raises(ValidationError):
            BlockedModel(blocked="anything")

        true_arm_schema = {
            "type": "object",
            "properties": {
                "permissive": {
                    "anyOf": [True, {"type": "string", "enum": ["read"]}],
                },
            },
            "required": ["permissive"],
        }
        TrueModel = create_model_from_schema(true_arm_schema)
        assert TrueModel(permissive="arbitrary").permissive == "arbitrary"

    def test_all_of_typeless_scalar_constraints_enforced(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "val": {
                    "allOf": [
                        {"pattern": "^ok_[0-9]+$"},
                        {"minLength": 4},
                    ]
                }
            },
            "required": ["val"],
        }
        Model = create_model_from_schema(schema)
        assert Model(val="ok_1").val == "ok_1"
        for bad in ("bad", "ok_"):
            with pytest.raises(ValidationError):
                Model(val=bad)

    def test_single_member_all_of_array_bounds_enforced_in_items(self) -> None:
        schema = {
            "type": "object",
            "$defs": {"Row": {"type": "array", "items": {"type": "string"}}},
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "minItems": 1,
                        "maxItems": 1,
                        "allOf": [{"$ref": "#/$defs/Row"}],
                    },
                }
            },
            "required": ["matrix"],
        }
        Model = create_model_from_schema(schema)
        assert Model(matrix=[["ok"]]).matrix == [["ok"]]
        for bad in ([["a", "b"]], [[]]):
            with pytest.raises(ValidationError):
                Model(matrix=bad)

    def test_single_member_all_of_array_bounds_enforced_in_any_of(self) -> None:
        schema = {
            "type": "object",
            "$defs": {"Row": {"type": "array", "items": {"type": "string"}}},
            "properties": {
                "payload": {
                    "anyOf": [
                        {
                            "minItems": 1,
                            "maxItems": 1,
                            "allOf": [{"$ref": "#/$defs/Row"}],
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["payload"],
        }
        Model = create_model_from_schema(schema)
        assert Model(payload=["ok"]).payload == ["ok"]
        assert Model(payload=None).payload is None
        for bad in (["a", "b"], []):
            with pytest.raises(ValidationError):
                Model(payload=bad)

    def test_all_of_uri_format_enforces_string_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "endpoint": {
                    "allOf": [
                        {"type": "string", "format": "uri"},
                        {"pattern": r"^https://allowed\.example\.com/"},
                        {"minLength": 32},
                    ]
                }
            },
            "required": ["endpoint"],
        }
        Model = create_model_from_schema(schema)
        assert Model.model_json_schema()
        ok = Model(endpoint="https://allowed.example.com/api/v1")
        assert isinstance(ok.endpoint, AnyUrl)
        assert isinstance(Model(endpoint=ok.endpoint).endpoint, AnyUrl)
        for bad in (
            "https://evil.example.com/api/v1",
            AnyUrl("https://evil.example.com/api/v1"),
            "https://allowed.example.com/x",
            AnyUrl("https://allowed.example.com/x"),
        ):
            with pytest.raises(ValidationError):
                Model(endpoint=bad)

    def test_all_of_file_path_and_password_formats_enforce_string_constraints(
        self, tmp_path: Path
    ) -> None:
        valid_file = tmp_path / "allowed_report.txt"
        short_file = tmp_path / "allowed_a.txt"
        blocked_file = tmp_path / "blocked_report.txt"
        for path in (valid_file, short_file, blocked_file):
            path.write_text("ok", encoding="utf-8")

        schema = {
            "type": "object",
            "properties": {
                "report": {
                    "allOf": [
                        {"type": "string", "format": "file-path"},
                        {"pattern": r"(^|[/\\])allowed_[^/\\]+\.txt$"},
                        {"minLength": len(str(valid_file))},
                    ]
                },
                "token": {
                    "allOf": [
                        {"type": "string", "format": "password"},
                        {"pattern": r"^tok_"},
                        {"minLength": 8},
                    ]
                },
            },
            "required": ["report", "token"],
        }
        Model = create_model_from_schema(schema)
        assert Model.model_json_schema()
        ok = Model(report=str(valid_file), token="tok_1234")
        assert ok.report == valid_file
        assert isinstance(ok.token, SecretStr)
        assert ok.token.get_secret_value() == "tok_1234"
        assert Model(report=valid_file, token=SecretStr("tok_1234")).report == valid_file

        for bad_report in (str(blocked_file), blocked_file, str(short_file), short_file):
            with pytest.raises(ValidationError):
                Model(report=bad_report, token="tok_1234")
        for bad_token in ("bad_1234", "tok_1", SecretStr("bad_1234"), SecretStr("tok_1")):
            with pytest.raises(ValidationError):
                Model(report=str(valid_file), token=bad_token)


# $ref resolution


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


# model_name parameter


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


# Property names that collide with BaseModel attributes
#
# Pydantic warns that such a field "shadows an attribute in parent BaseModel". The
# warning is cosmetic and is left on: the field keeps its own name everywhere, which
# is what tools, flow state restore and structured output read.


class TestBaseModelAttributeNames:
    @pytest.mark.parametrize("name", ["schema", "json", "copy"])
    def test_property_keeps_its_name(self, name: str) -> None:
        schema = {
            "type": "object",
            "title": "ActionRequest",
            "properties": {name: {"type": "string"}},
            "required": [name],
        }
        Model = create_model_from_schema(schema)
        instance = Model(**{name: "value"})

        assert list(Model.model_fields) == [name]
        assert getattr(instance, name) == "value"
        assert instance.model_dump() == {name: "value"}
        assert list(Model.model_json_schema()["properties"]) == [name]

    def test_nested_object_property_keeps_its_name(self) -> None:
        schema = {
            "type": "object",
            "title": "CreateList",
            "properties": {
                "list": {
                    "type": "object",
                    "properties": {"schema": {"type": "string"}},
                    "required": ["schema"],
                }
            },
            "required": ["list"],
        }
        Model = create_model_from_schema(schema)

        dumped = Model(list={"schema": "columns"}).model_dump()
        assert dumped == {"list": {"schema": "columns"}}

    def test_shadowing_property_is_still_validated(self) -> None:
        schema = {
            "type": "object",
            "title": "ActionRequest",
            "properties": {"schema": {"type": "integer"}},
            "required": ["schema"],
        }
        Model = create_model_from_schema(schema)

        with pytest.raises(ValueError, match="schema\n  Field required"):
            Model()
        with pytest.raises(ValueError, match="schema\n  Input should be a valid integer"):
            Model(schema="not a number")

    def test_recursive_defs_keep_shadowing_names(self) -> None:
        # B refers back to A, so it is built incomplete and finished by model_rebuild.
        schema = {
            "type": "object",
            "title": "Root",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {
                        "schema": {"type": "string"},
                        "b": {"$ref": "#/$defs/B"},
                    },
                },
                "B": {
                    "type": "object",
                    "properties": {
                        "json": {"type": "string"},
                        "a": {"$ref": "#/$defs/A"},
                    },
                },
            },
        }
        Model = create_model_from_schema(schema)

        dumped = Model(a={"schema": "s", "b": {"json": "j"}}).model_dump()
        assert dumped["a"]["schema"] == "s"
        assert dumped["a"]["b"]["json"] == "j"


# enrich_descriptions


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


# Edge cases


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


# build_rich_field_description


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


# Schema transformation functions


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


# Recursive / circular $ref schemas (GH-5490)

RECURSIVE_NODE_SCHEMA: dict = {
    "$defs": {
        "Node": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Node"},
                },
            },
            "required": ["name"],
        }
    },
    "$ref": "#/$defs/Node",
}

MUTUAL_RECURSION_SCHEMA: dict = {
    "$defs": {
        "A": {
            "type": "object",
            "properties": {
                "val": {"type": "string"},
                "b": {"$ref": "#/$defs/B"},
            },
            "required": ["val"],
        },
        "B": {
            "type": "object",
            "properties": {
                "val": {"type": "integer"},
                "a": {"$ref": "#/$defs/A"},
            },
            "required": ["val"],
        },
    },
    "$ref": "#/$defs/A",
}


class TestResolveRefsRecursive:
    def test_circular_ref_preserves_type(self) -> None:
        from crewai.utilities.pydantic_schema_utils import resolve_refs

        resolved = resolve_refs(deepcopy(RECURSIVE_NODE_SCHEMA))
        items = resolved["properties"]["children"]["items"]
        assert items != {}, "Circular ref should not degrade to {}"
        assert items.get("type") == "object"

    def test_non_recursive_schema_still_resolves(self) -> None:
        from crewai.utilities.pydantic_schema_utils import resolve_refs

        schema = {
            "$defs": {"Foo": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "$ref": "#/$defs/Foo",
        }
        resolved = resolve_refs(schema)
        assert resolved["properties"]["x"]["type"] == "integer"


class TestSanitizeStrictSchemas:
    def test_openai_strict_preserves_property_named_title(self) -> None:
        from crewai.utilities.pydantic_schema_utils import sanitize_tool_params_for_openai_strict

        schema = {
            "type": "object",
            "properties": {
                "title": {"title": "Title", "type": "string"},
                "url": {"title": "Url", "type": "string"},
            },
            "required": ["title", "url"],
        }

        san = sanitize_tool_params_for_openai_strict(deepcopy(schema))

        assert "title" in san["properties"]
        assert set(san["required"]) == set(san["properties"].keys())
        assert "title" not in san["properties"]["title"]

    def test_openai_strict_preserves_nested_property_named_title(self) -> None:
        from crewai.utilities.pydantic_schema_utils import sanitize_tool_params_for_openai_strict

        schema = {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {
                        "title": {"title": "Nested Title", "type": "string"},
                    },
                    "required": ["title"],
                },
            },
            "required": ["payload"],
        }

        san = sanitize_tool_params_for_openai_strict(deepcopy(schema))
        payload = san["properties"]["payload"]

        assert "title" in payload["properties"]
        assert payload["required"] == ["title"]
        assert "title" not in payload["properties"]["title"]

    def test_anthropic_strict_preserves_recursive_type(self) -> None:
        from crewai.utilities.pydantic_schema_utils import sanitize_tool_params_for_anthropic_strict

        san = sanitize_tool_params_for_anthropic_strict(deepcopy(RECURSIVE_NODE_SCHEMA))
        items = san["properties"]["children"]["items"]
        assert items != {}
        assert items.get("type") == "object"

    def test_openai_strict_preserves_recursive_type(self) -> None:
        from crewai.utilities.pydantic_schema_utils import sanitize_tool_params_for_openai_strict

        san = sanitize_tool_params_for_openai_strict(deepcopy(RECURSIVE_NODE_SCHEMA))
        items = san["properties"]["children"]["items"]
        assert items != {}
        assert items.get("type") == "object"


class TestCreateModelFromSchemaRecursive:
    def test_model_creation_succeeds(self) -> None:
        model = create_model_from_schema(deepcopy(RECURSIVE_NODE_SCHEMA), model_name="Node")
        assert model is not None
        assert model.__name__ == "Node"

    def test_model_accepts_valid_recursive_data(self) -> None:
        model = create_model_from_schema(deepcopy(RECURSIVE_NODE_SCHEMA), model_name="Node")
        instance = model(name="root", children=[{"name": "child", "children": []}])
        assert instance.name == "root"
        assert len(instance.children) == 1

    def test_model_rejects_missing_required_field(self) -> None:
        import pytest

        model = create_model_from_schema(deepcopy(RECURSIVE_NODE_SCHEMA), model_name="Node")
        with pytest.raises(Exception):
            model(children=[])

    def test_mutual_recursion_schema(self) -> None:
        model = create_model_from_schema(deepcopy(MUTUAL_RECURSION_SCHEMA), model_name="A")
        instance = model(val="hello", b={"val": 42})
        assert instance.val == "hello"

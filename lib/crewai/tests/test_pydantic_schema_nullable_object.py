"""
Regression tests for crewAI issue #7908:
create_model_from_schema must produce Optional[Model] for a field declared with
{"type": ["object", "null"], "properties": {...}}, regardless of member order,
number of sibling fields, and whether the field is required.

Root cause: _json_schema_to_pydantic_type created temporary dicts
{**json_schema, "type": member} for each list-form type member. These dicts
were registered in in_progress by id(). After the function returned, the dicts
were GC'd. Python could reuse their id()s for later temporary dicts, causing
in_progress to return the wrong cached model.

Fix: a _schema_keepalive list is created in create_model_from_schema and
threaded through the entire conversion. Every temporary schema dict is appended
to it, keeping all dicts alive for the full duration of the top-level conversion.
"""
import pytest
from pydantic import ValidationError

from crewai.utilities.pydantic_schema_utils import create_model_from_schema


@pytest.mark.parametrize("type_order", [
    ["object", "null"],
    ["null", "object"],
])
def test_nullable_nested_object_accepts_none(type_order):
    """A field with type [object, null] must accept None regardless of member order."""
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": type_order,
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    Model = create_model_from_schema(schema)
    instance = Model(payload=None)
    assert instance.payload is None


@pytest.mark.parametrize("type_order", [
    ["object", "null"],
    ["null", "object"],
])
def test_nullable_nested_object_accepts_object(type_order):
    """A field with type [object, null] must also accept the object form."""
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": type_order,
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    Model = create_model_from_schema(schema)
    instance = Model(payload={"value": "hello"})
    assert instance.payload is not None


@pytest.mark.parametrize("type_order", [
    ["object", "null"],
    ["null", "object"],
])
def test_required_nullable_field_accepts_none(type_order):
    """A required field with type [object, null] must accept None."""
    schema = {
        "type": "object",
        "required": ["payload"],
        "properties": {
            "payload": {
                "type": type_order,
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        },
    }
    Model = create_model_from_schema(schema)
    instance = Model(payload=None)
    assert instance.payload is None


def test_two_sibling_nullable_object_fields_get_distinct_models():
    """Two nullable-object fields with different properties must produce distinct models.

    This is the multi-field regression case: if member_schema dicts are GC'd
    between conversions, a later field's schema can reuse the first field's
    id() in in_progress, returning the wrong model for the second field.
    """
    schema = {
        "type": "object",
        "required": ["alpha", "beta"],
        "properties": {
            "alpha": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "beta": {
                "type": ["object", "null"],
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            },
        },
    }
    Model = create_model_from_schema(schema)
    assert Model(alpha=None, beta=None).alpha is None
    assert Model(alpha=None, beta=None).beta is None
    inst = Model(alpha={"x": "hello"}, beta={"y": 42})
    assert inst.alpha is not None
    assert inst.beta is not None


def test_n_sibling_nullable_object_fields(n: int = 20):
    """N sibling nullable-object fields must each accept None and their own schema.

    This is VANDRANKI's stress test from the PR review. With N=20 and
    type_order=[object, null], the original fix still produced wrong-model
    results because member_schemas only lived until the type-array conversion
    returned. The keepalive list fixes this for all N.
    """
    props = {
        f"f{k}": {
            "type": ["object", "null"],
            "properties": {f"k{k}": {"type": "string"}},
            "required": [f"k{k}"],
        }
        for k in range(n)
    }
    schema = {"type": "object", "required": list(props.keys()), "properties": props}
    Model = create_model_from_schema(schema)

    # All fields accept None
    all_none = {f"f{k}": None for k in range(n)}
    inst = Model(**all_none)
    for k in range(n):
        assert getattr(inst, f"f{k}") is None, f"f{k} should accept None"

    # Each field accepts its own schema
    for k in range(n):
        kwargs = {f"f{j}": None for j in range(n)}
        kwargs[f"f{k}"] = {f"k{k}": "test"}
        inst2 = Model(**kwargs)
        assert getattr(inst2, f"f{k}") is not None, f"f{k} should accept its own schema"


@pytest.mark.parametrize("type_order", [
    ["object", "null"],
    ["null", "object"],
])
def test_nullable_nested_object_rejects_wrong_type(type_order):
    """A field with type [object, null] must reject non-null non-object values."""
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": type_order,
                "properties": {"value": {"type": "string"}},
            }
        },
    }
    Model = create_model_from_schema(schema)
    with pytest.raises(ValidationError):
        Model(payload="not-an-object-or-null")

"""Dynamic Pydantic model creation from JSON schemas.

This module provides utilities for converting JSON schemas to Pydantic models at runtime.
The main function is `create_model_from_schema`, which takes a JSON schema and returns
a dynamically created Pydantic model class.

This is used by the A2A server to honor response schemas sent by clients, allowing
structured output from agent tasks.

Based on dydantic (https://github.com/zenbase-ai/dydantic).

This module provides functions for converting Pydantic models to JSON schemas
suitable for use with LLMs and tool definitions.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import datetime
import ipaddress
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    ForwardRef,
    Literal,
    Optional,
    TypeGuard,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
)
import uuid

import jsonref  # type: ignore[import-untyped]
from pydantic import (
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    AfterValidator,
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    FileUrl,
    HttpUrl,
    Json,
    MongoDsn,
    NewPath,
    PostgresDsn,
    SecretBytes,
    SecretStr,
    StrictBytes,
    TypeAdapter,
    ValidationError,
    create_model as create_model_base,
    model_validator,
)
from pydantic.networks import (  # type: ignore[attr-defined]
    IPv4Address,
    IPv6Address,
    IPvAnyAddress,
    IPvAnyInterface,
    IPvAnyNetwork,
)
from pydantic_core import SchemaError


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic import EmailStr
    from pydantic.main import AnyClassMethod
else:
    try:
        from pydantic import EmailStr
    except ImportError:
        logger.warning(
            "EmailStr unavailable, using str fallback",
            extra={"missing_package": "email_validator"},
        )
        EmailStr = str


class JsonSchemaInfo(TypedDict):
    """Inner structure for JSON schema metadata."""

    name: str
    strict: Literal[True]
    schema: dict[str, Any]


class ModelDescription(TypedDict):
    """Return type for generate_model_description."""

    type: Literal["json_schema"]
    json_schema: JsonSchemaInfo


def resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve all local $refs in the given JSON Schema using $defs as the source.

    This is needed because Pydantic generates $ref-based schemas that
    some consumers (e.g. LLMs, tool frameworks) don't handle well.

    Args:
        schema: JSON Schema dict that may contain "$refs" and "$defs".

    Returns:
        A new schema dictionary with all local $refs replaced by their definitions.
    """
    defs = schema.get("$defs", {})
    schema_copy = deepcopy(schema)
    expanding: set[str] = set()

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                if def_name not in defs:
                    raise KeyError(f"Definition '{def_name}' not found in $defs.")
                if def_name in expanding:
                    def_schema = defs[def_name]
                    stub: dict[str, Any] = {"type": def_schema.get("type", "object")}
                    if "description" in def_schema:
                        stub["description"] = def_schema["description"]
                    return stub
                expanding.add(def_name)
                try:
                    return _resolve(deepcopy(defs[def_name]))
                finally:
                    expanding.discard(def_name)
            return {k: _resolve(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_resolve(i) for i in node]

        return node

    return _resolve(schema_copy)  # type: ignore[no-any-return]


def add_key_in_dict_recursively(
    d: dict[str, Any],
    key: str,
    value: Any,
    criteria: Callable[[dict[str, Any]], bool],
    _seen: set[int] | None = None,
) -> dict[str, Any]:
    """Recursively adds a key/value pair to all nested dicts matching `criteria`.

    Args:
        d: The dictionary to modify.
        key: The key to add.
        value: The value to add.
        criteria: A function that returns True for dicts that should receive the key.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        The modified dictionary.
    """
    if _seen is None:
        _seen = set()
    if isinstance(d, dict):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        if criteria(d) and key not in d:
            d[key] = value
        for v in d.values():
            add_key_in_dict_recursively(v, key, value, criteria, _seen)
    elif isinstance(d, list):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for i in d:
            add_key_in_dict_recursively(i, key, value, criteria, _seen)
    return d


def force_additional_properties_false(d: Any, _seen: set[int] | None = None) -> Any:
    """Force additionalProperties=false on all object-type dicts recursively.

    OpenAI strict mode requires all objects to have additionalProperties=false.
    This function overwrites any existing value to ensure compliance.

    Also ensures objects have properties and required arrays, even if empty,
    as OpenAI strict mode requires these for all object types.

    Args:
        d: The dictionary/list to modify.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        The modified dictionary/list.
    """
    if _seen is None:
        _seen = set()
    if isinstance(d, dict):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        if d.get("type") == "object":
            d["additionalProperties"] = False
            if "properties" not in d:
                d["properties"] = {}
            if "required" not in d:
                d["required"] = []
        for v in d.values():
            force_additional_properties_false(v, _seen)
    elif isinstance(d, list):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for i in d:
            force_additional_properties_false(i, _seen)
    return d


OPENAI_SUPPORTED_FORMATS: Final[
    set[Literal["date-time", "date", "time", "duration"]]
] = {
    "date-time",
    "date",
    "time",
    "duration",
}


def strip_unsupported_formats(d: Any, _seen: set[int] | None = None) -> Any:
    """Remove format annotations that OpenAI strict mode doesn't support.

    OpenAI only supports: date-time, date, time, duration.
    Other formats like uri, email, uuid etc. cause validation errors.

    Args:
        d: The dictionary/list to modify.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        The modified dictionary/list.
    """
    if _seen is None:
        _seen = set()
    if isinstance(d, dict):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        format_value = d.get("format")
        if (
            isinstance(format_value, str)
            and format_value not in OPENAI_SUPPORTED_FORMATS
        ):
            del d["format"]
        for v in d.values():
            strip_unsupported_formats(v, _seen)
    elif isinstance(d, list):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for i in d:
            strip_unsupported_formats(i, _seen)
    return d


def ensure_type_in_schemas(d: Any, _seen: set[int] | None = None) -> Any:
    """Ensure all schema objects in anyOf/oneOf have a 'type' key.

    OpenAI strict mode requires every schema to have a 'type' key.
    Empty schemas {} in anyOf/oneOf are converted to {"type": "object"}.

    Args:
        d: The dictionary/list to modify.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        The modified dictionary/list.
    """
    if _seen is None:
        _seen = set()
    if isinstance(d, dict):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for key in ("anyOf", "oneOf"):
            if key in d:
                schema_list = d[key]
                for i, schema in enumerate(schema_list):
                    if isinstance(schema, dict) and schema == {}:
                        schema_list[i] = {"type": "object"}
                    else:
                        ensure_type_in_schemas(schema, _seen)
        for v in d.values():
            ensure_type_in_schemas(v, _seen)
    elif isinstance(d, list):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for item in d:
            ensure_type_in_schemas(item, _seen)
    return d


def fix_discriminator_mappings(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace '#/$defs/...' references in discriminator.mapping with just the model name.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Modified schema with fixed discriminator mappings.
    """
    output = schema.get("properties", {}).get("output")
    if not output:
        return schema

    disc = output.get("discriminator")
    if not disc or "mapping" not in disc:
        return schema

    disc["mapping"] = {k: v.split("/")[-1] for k, v in disc["mapping"].items()}
    return schema


def add_const_to_oneof_variants(schema: dict[str, Any]) -> dict[str, Any]:
    """Add const fields to oneOf variants for discriminated unions.

    The json_schema_to_pydantic library requires each oneOf variant to have
    a const field for the discriminator property. This function adds those
    const fields based on the discriminator mapping.

    Args:
        schema: JSON Schema dict that may contain discriminated unions

    Returns:
        Modified schema with const fields added to oneOf variants
    """

    def _process_oneof(node: dict[str, Any]) -> dict[str, Any]:
        """Process a single node that might contain a oneOf with discriminator."""
        if not isinstance(node, dict):
            return node

        if "oneOf" in node and "discriminator" in node:
            discriminator = node["discriminator"]
            property_name = discriminator.get("propertyName")
            mapping = discriminator.get("mapping", {})

            if property_name and mapping:
                one_of_variants = node.get("oneOf", [])

                for variant in one_of_variants:
                    if isinstance(variant, dict) and "properties" in variant:
                        variant_title = variant.get("title", "")

                        matched_disc_value = None
                        for disc_value, schema_name in mapping.items():
                            if variant_title == schema_name or variant_title.endswith(
                                schema_name
                            ):
                                matched_disc_value = disc_value
                                break

                        if matched_disc_value is not None:
                            props = variant["properties"]
                            if property_name in props:
                                props[property_name]["const"] = matched_disc_value

        for key, value in node.items():
            if isinstance(value, dict):
                node[key] = _process_oneof(value)
            elif isinstance(value, list):
                node[key] = [
                    _process_oneof(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return node

    return _process_oneof(deepcopy(schema))


def convert_oneof_to_anyof(
    schema: dict[str, Any], _seen: set[int] | None = None
) -> dict[str, Any]:
    """Convert oneOf to anyOf for OpenAI compatibility.

    OpenAI's Structured Outputs support anyOf better than oneOf.
    This recursively converts all oneOf occurrences to anyOf.

    Args:
        schema: JSON schema dictionary.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        Modified schema with anyOf instead of oneOf.
    """
    if _seen is None:
        _seen = set()
    if isinstance(schema, dict):
        if id(schema) in _seen:
            return schema
        _seen.add(id(schema))
        if "oneOf" in schema:
            schema["anyOf"] = schema.pop("oneOf")

        for value in schema.values():
            if isinstance(value, dict):
                convert_oneof_to_anyof(value, _seen)
            elif isinstance(value, list):
                if id(value) in _seen:
                    continue
                _seen.add(id(value))
                for item in value:
                    if isinstance(item, dict):
                        convert_oneof_to_anyof(item, _seen)

    return schema


def ensure_all_properties_required(
    schema: dict[str, Any], _seen: set[int] | None = None
) -> dict[str, Any]:
    """Ensure all properties are in the required array for OpenAI strict mode.

    OpenAI's strict structured outputs require all properties to be listed
    in the required array. This recursively updates all objects to include
    all their properties in required.

    Args:
        schema: JSON schema dictionary.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        Modified schema with all properties marked as required.
    """
    if _seen is None:
        _seen = set()
    if isinstance(schema, dict):
        if id(schema) in _seen:
            return schema
        _seen.add(id(schema))
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema["properties"]
            if properties:
                schema["required"] = list(properties.keys())

        for value in schema.values():
            if isinstance(value, dict):
                ensure_all_properties_required(value, _seen)
            elif isinstance(value, list):
                if id(value) in _seen:
                    continue
                _seen.add(id(value))
                for item in value:
                    if isinstance(item, dict):
                        ensure_all_properties_required(item, _seen)

    return schema


def strip_null_from_types(
    schema: dict[str, Any], _seen: set[int] | None = None
) -> dict[str, Any]:
    """Remove null type from anyOf/type arrays.

    Pydantic generates `T | None` for optional fields, which creates schemas with
    null in the type. However, for MCP tools, optional fields should be omitted
    entirely rather than sent as null. This function strips null from types.

    Args:
        schema: JSON schema dictionary.
        _seen: Internal set of visited ``id()``s, used to guard cyclic schemas.

    Returns:
        Modified schema with null types removed.
    """
    if _seen is None:
        _seen = set()
    if isinstance(schema, dict):
        if id(schema) in _seen:
            return schema
        _seen.add(id(schema))
        if "anyOf" in schema:
            any_of = schema["anyOf"]
            non_null = [opt for opt in any_of if opt.get("type") != "null"]
            if len(non_null) == 1:
                schema.pop("anyOf")
                schema.update(non_null[0])
            elif len(non_null) > 1:
                schema["anyOf"] = non_null

        type_value = schema.get("type")
        if isinstance(type_value, list) and "null" in type_value:
            non_null_types = [t for t in type_value if t != "null"]
            if len(non_null_types) == 1:
                schema["type"] = non_null_types[0]
            elif len(non_null_types) > 1:
                schema["type"] = non_null_types

        for value in schema.values():
            if isinstance(value, dict):
                strip_null_from_types(value, _seen)
            elif isinstance(value, list):
                if id(value) in _seen:
                    continue
                _seen.add(id(value))
                for item in value:
                    if isinstance(item, dict):
                        strip_null_from_types(item, _seen)

    return schema


_STRICT_METADATA_KEYS: Final[tuple[str, ...]] = (
    "title",
    "default",
    "examples",
    "example",
    "$comment",
    "readOnly",
    "writeOnly",
    "deprecated",
)

_CLAUDE_STRICT_UNSUPPORTED: Final[tuple[str, ...]] = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
)


def _strip_keys_recursive(
    d: Any, keys: tuple[str, ...], _seen: set[int] | None = None
) -> Any:
    """Recursively delete schema metadata keys without deleting property names.

    JSON Schema stores fields under a ``properties`` map. A user/tool argument can
    legitimately be named ``title``, ``default``, or another metadata key. Those
    entries must stay in the map; only metadata inside the schema nodes should be
    stripped.
    """
    if _seen is None:
        _seen = set()
    if isinstance(d, dict):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for key in keys:
            d.pop(key, None)
        for key, value in d.items():
            if key == "properties" and isinstance(value, dict):
                for property_schema in value.values():
                    _strip_keys_recursive(property_schema, keys, _seen)
            else:
                _strip_keys_recursive(value, keys, _seen)
    elif isinstance(d, list):
        if id(d) in _seen:
            return d
        _seen.add(id(d))
        for i in d:
            _strip_keys_recursive(i, keys, _seen)
    return d


def lift_top_level_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a top-level anyOf/oneOf/allOf wrapping a single object variant.

    Anthropic's strict ``input_schema`` rejects top-level union keywords. When
    exactly one variant is an object schema, lift it so the root is a plain
    object; otherwise leave the schema alone.
    """
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if not isinstance(variants, list):
            continue
        object_variants = [
            v for v in variants if isinstance(v, dict) and v.get("type") == "object"
        ]
        if len(object_variants) == 1:
            lifted = deepcopy(object_variants[0])
            schema.pop(key)
            schema.update(lifted)
            break
    return schema


def _common_strict_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    """Shared strict sanitization: inline refs, close objects, require all properties."""
    sanitized = resolve_refs(deepcopy(params))
    sanitized.pop("$defs", None)
    sanitized = convert_oneof_to_anyof(sanitized)
    sanitized = ensure_type_in_schemas(sanitized)
    sanitized = force_additional_properties_false(sanitized)
    sanitized = ensure_all_properties_required(sanitized)
    return cast(dict[str, Any], _strip_keys_recursive(sanitized, _STRICT_METADATA_KEYS))


def sanitize_tool_params_for_openai_strict(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize a JSON schema for OpenAI strict function calling."""
    if not isinstance(params, dict):
        return params
    return cast(
        dict[str, Any], strip_unsupported_formats(_common_strict_pipeline(params))
    )


def sanitize_tool_params_for_anthropic_strict(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize a JSON schema for Anthropic strict tool use."""
    if not isinstance(params, dict):
        return params
    sanitized = lift_top_level_anyof(_common_strict_pipeline(params))
    sanitized = _strip_keys_recursive(sanitized, _CLAUDE_STRICT_UNSUPPORTED)
    return cast(dict[str, Any], strip_unsupported_formats(sanitized))


def sanitize_tool_params_for_bedrock_strict(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize a JSON schema for Bedrock Converse strict tool use.

    Bedrock Converse uses the same grammar compiler as the underlying Claude
    model, so the constraints match Anthropic's.
    """
    return sanitize_tool_params_for_anthropic_strict(params)


def generate_model_description(
    model: type[BaseModel],
    *,
    strip_null_types: bool = True,
) -> ModelDescription:
    """Generate JSON schema description of a Pydantic model.

    This function takes a Pydantic model class and returns its JSON schema,
    which includes full type information, discriminators, and all metadata.
    The schema is dereferenced to inline all $ref references for better LLM understanding.

    Args:
        model: A Pydantic model class.
        strip_null_types: When ``True`` (default), remove ``null`` from
            ``anyOf`` / ``type`` arrays.  Set to ``False`` to allow sending ``null`` for
            optional fields.

    Returns:
        A ModelDescription with JSON schema representation of the model.
    """
    json_schema = model.model_json_schema(ref_template="#/$defs/{model}")

    json_schema = force_additional_properties_false(json_schema)
    json_schema = strip_unsupported_formats(json_schema)
    json_schema = ensure_type_in_schemas(json_schema)

    json_schema = resolve_refs(json_schema)

    json_schema.pop("$defs", None)
    json_schema = fix_discriminator_mappings(json_schema)
    json_schema = convert_oneof_to_anyof(json_schema)
    json_schema = ensure_all_properties_required(json_schema)

    if strip_null_types:
        json_schema = strip_null_from_types(json_schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": json_schema,
        },
    }


FORMAT_TYPE_MAP: dict[str, type[Any]] = {
    "base64": Annotated[bytes, Field(json_schema_extra={"format": "base64"})],  # type: ignore[dict-item]
    "binary": StrictBytes,
    "date": datetime.date,
    "time": datetime.time,
    "date-time": datetime.datetime,
    "duration": datetime.timedelta,
    "directory-path": DirectoryPath,
    "email": EmailStr,
    "file-path": FilePath,
    "ipv4": IPv4Address,
    "ipv6": IPv6Address,
    "ipvanyaddress": IPvAnyAddress,  # type: ignore[dict-item]
    "ipvanyinterface": IPvAnyInterface,  # type: ignore[dict-item]
    "ipvanynetwork": IPvAnyNetwork,  # type: ignore[dict-item]
    "json-string": Json,
    "multi-host-uri": PostgresDsn | MongoDsn,  # type: ignore[dict-item]
    "password": SecretStr,
    "path": NewPath,
    "uri": AnyUrl,
    "uuid": uuid.UUID,
    "uuid1": UUID1,
    "uuid3": UUID3,
    "uuid4": UUID4,
    "uuid5": UUID5,
}


def build_rich_field_description(prop_schema: dict[str, Any]) -> str:
    """Build a comprehensive field description including constraints.

    Embeds format, enum, pattern, min/max, and example constraints into the
    description text so that LLMs can understand tool parameter requirements
    without inspecting the raw JSON Schema.

    Args:
        prop_schema: Property schema with description and constraints.

    Returns:
        Enhanced description with format, enum, and other constraints.
    """
    parts: list[str] = []

    description = prop_schema.get("description", "")
    if description:
        parts.append(description)

    format_type = prop_schema.get("format")
    if format_type:
        parts.append(f"Format: {format_type}")

    enum_values = prop_schema.get("enum")
    if enum_values:
        enum_str = ", ".join(repr(v) for v in enum_values)
        parts.append(f"Allowed values: [{enum_str}]")

    pattern = prop_schema.get("pattern")
    if pattern:
        parts.append(f"Pattern: {pattern}")

    minimum = prop_schema.get("minimum")
    maximum = prop_schema.get("maximum")
    if minimum is not None:
        parts.append(f"Minimum: {minimum}")
    if maximum is not None:
        parts.append(f"Maximum: {maximum}")

    min_length = prop_schema.get("minLength")
    max_length = prop_schema.get("maxLength")
    if min_length is not None:
        parts.append(f"Min length: {min_length}")
    if max_length is not None:
        parts.append(f"Max length: {max_length}")

    examples = prop_schema.get("examples")
    if examples:
        examples_str = ", ".join(repr(e) for e in examples[:3])
        parts.append(f"Examples: {examples_str}")

    return ". ".join(parts) if parts else ""


def _inline_top_level_ref(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve only the top-level ``$ref``, preserving ``$defs`` for lazy inner resolution.

    Used as a fallback when ``jsonref.replace_refs`` fails on circular schemas.
    Inner ``$ref`` pointers are left intact so that :func:`_resolve_ref` can
    resolve them during model construction, with cycle detection via ``in_progress``.
    """
    schema = deepcopy(schema)
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        def_name = ref[len("#/$defs/") :]
        defs = schema.get("$defs", {})
        if def_name in defs:
            resolved: dict[str, Any] = deepcopy(defs[def_name])
            resolved.setdefault("$defs", defs)
            return resolved
    return schema


def serialize_model_class(value: Any) -> Any:
    """Serialize a ``type[BaseModel]`` field value as its JSON schema.

    Args:
        value: A ``type[BaseModel]`` subclass, ``None``, or another union member.

    Returns:
        ``value.model_json_schema()`` for model classes, ``value`` otherwise.
    """
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema()
    return value


def create_model_from_schema(  # type: ignore[no-any-unimported]
    json_schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
    model_name: str | None = None,
    enrich_descriptions: bool = False,
    __config__: ConfigDict | None = None,
    __base__: type[BaseModel] | None = None,
    __module__: str = __name__,
    __validators__: dict[str, AnyClassMethod] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
) -> type[BaseModel]:
    """Create a Pydantic model from a JSON schema.

    This function takes a JSON schema as input and dynamically creates a Pydantic
    model class based on the schema. It supports various JSON schema features such
    as nested objects, referenced definitions ($ref), arrays with typed items,
    union types (anyOf/oneOf), and string formats.

    Args:
        json_schema: A dictionary representing the JSON schema.
        root_schema: The root schema containing $defs. If not provided, the
            current schema is treated as the root schema.
        model_name: Override for the model name. If not provided, the schema
            ``title`` field is used, falling back to ``"DynamicModel"``.
        enrich_descriptions: When True, augment field descriptions with
            constraint info (format, enum, pattern, min/max, examples) via
            :func:`build_rich_field_description`.  Useful for LLM-facing tool
            schemas where constraints in the description help the model
            understand parameter requirements.
        __config__: Pydantic configuration for the generated model.
        __base__: Base class for the generated model. Defaults to BaseModel.
        __module__: Module name for the generated model class.
        __validators__: A dictionary of custom validators for the generated model.
        __cls_kwargs__: Additional keyword arguments for the generated model class.

    Returns:
        A dynamically created Pydantic model class based on the provided JSON schema.

    Example:
        >>> schema = {
        ...     "title": "Person",
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"},
        ...     },
        ...     "required": ["name"],
        ... }
        >>> Person = create_model_from_schema(schema)
        >>> person = Person(name="John", age=30)
        >>> person.name
        'John'
    """
    try:
        json_schema = dict(jsonref.replace_refs(json_schema, proxies=False))
    except (jsonref.JsonRefError, RecursionError):
        json_schema = _inline_top_level_ref(json_schema)

    effective_root = root_schema or json_schema

    json_schema = force_additional_properties_false(json_schema)
    effective_root = force_additional_properties_false(effective_root)

    in_progress: dict[int, Any] = {}
    model = _build_model_from_schema(
        json_schema,
        effective_root,
        model_name=model_name,
        enrich_descriptions=enrich_descriptions,
        in_progress=in_progress,
        __config__=__config__,
        __base__=__base__,
        __module__=__module__,
        __validators__=__validators__,
        __cls_kwargs__=__cls_kwargs__,
    )

    types_namespace: dict[str, Any] = {
        entry.__name__: entry
        for entry in in_progress.values()
        if isinstance(entry, type) and issubclass(entry, BaseModel)
    }
    for entry in in_progress.values():
        if (
            isinstance(entry, type)
            and issubclass(entry, BaseModel)
            and not getattr(entry, "__pydantic_complete__", True)
        ):
            try:
                entry.model_rebuild(_types_namespace=types_namespace)
            except Exception as e:
                logger.debug("model_rebuild failed for %s: %s", entry.__name__, e)
    return model


def _build_model_from_schema(  # type: ignore[no-any-unimported]
    json_schema: dict[str, Any],
    effective_root: dict[str, Any],
    *,
    model_name: str | None,
    enrich_descriptions: bool,
    in_progress: dict[int, Any],
    __config__: ConfigDict | None = None,
    __base__: type[BaseModel] | None = None,
    __module__: str = __name__,
    __validators__: dict[str, AnyClassMethod] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
) -> type[BaseModel]:
    """Inner builder shared by the public entry point and recursive nested-object creation.

    Preprocessing via ``jsonref.replace_refs`` and the sanitization walkers is
    run once by the public entry; this helper walks the already-normalized
    schema and emits Pydantic models. ``in_progress`` maps ``id(schema)`` to
    the model being built for that schema, so a cyclic ``$ref`` graph
    degrades to a ``ForwardRef`` back-edge instead of blowing the stack.
    """
    original_id = id(json_schema)
    if json_schema.get("allOf"):
        json_schema = _merge_all_of_schemas(
            json_schema["allOf"],
            effective_root,
            outer_schema=json_schema,
        )

    effective_name = model_name or json_schema.get("title") or "DynamicModel"

    schema_id = id(json_schema)
    in_progress[original_id] = effective_name
    if schema_id != original_id:
        in_progress[schema_id] = effective_name

    field_definitions = {
        name: _json_schema_to_pydantic_field(
            name,
            prop,
            json_schema.get("required", []),
            effective_root,
            enrich_descriptions=enrich_descriptions,
            in_progress=in_progress,
        )
        for name, prop in (json_schema.get("properties", {}) or {}).items()
    }

    effective_config = __config__ or ConfigDict(extra="forbid")
    effective_validators: dict[str, Any] | None = __validators__
    if json_schema.get(_UNSATISFIABLE_KEY):
        effective_validators = {
            **(__validators__ or {}),
            "__crewai_unsatisfiable_validator__": model_validator(mode="before")(
                _reject_unsatisfiable_model
            ),
        }

    model = create_model_base(
        effective_name,
        __config__=effective_config,
        __base__=__base__,
        __module__=__module__,
        __validators__=effective_validators,
        __cls_kwargs__=__cls_kwargs__,
        **field_definitions,
    )
    in_progress[original_id] = model
    if schema_id != original_id:
        in_progress[schema_id] = model
    return model


def _json_schema_to_pydantic_field(
    name: str,
    json_schema: dict[str, Any],
    required: list[str],
    root_schema: dict[str, Any],
    *,
    enrich_descriptions: bool = False,
    in_progress: dict[int, Any] | None = None,
) -> Any:
    """Convert a JSON schema property to a Pydantic field definition.

    Args:
        name: The field name.
        json_schema: The JSON schema for this field.
        required: List of required field names.
        root_schema: The root schema for resolving $ref.
        enrich_descriptions: When True, embed constraints in the description.

    Returns:
        A tuple of (type, Field) for use with create_model.
    """
    type_ = _json_schema_to_pydantic_type(
        json_schema,
        root_schema,
        name_=name.title(),
        enrich_descriptions=enrich_descriptions,
        in_progress=in_progress,
    )
    from_all_of = bool(json_schema.get("allOf"))
    if from_all_of:
        json_schema = _merge_all_of_schemas(
            json_schema["allOf"],
            root_schema,
            outer_schema=json_schema,
        )
    if json_schema.get(_UNSATISFIABLE_KEY):
        type_ = Annotated[Any, AfterValidator(_reject_unsatisfiable)]
    is_required = name in required

    field_params: dict[str, Any] = {}
    schema_extra: dict[str, Any] = {}

    if enrich_descriptions:
        rich_desc = build_rich_field_description(json_schema)
        if rich_desc:
            field_params["description"] = rich_desc
    else:
        description = json_schema.get("description")
        if description:
            field_params["description"] = description

    examples = json_schema.get("examples")
    if examples:
        schema_extra["examples"] = examples

    has_default = "default" in json_schema
    default = json_schema["default"] if has_default else (... if is_required else None)

    applied_num_field_params = isinstance(type_, type) and issubclass(
        type_, (int, float)
    )
    if applied_num_field_params:
        if "minimum" in json_schema:
            field_params["ge"] = json_schema["minimum"]
        if "exclusiveMinimum" in json_schema:
            field_params["gt"] = json_schema["exclusiveMinimum"]
        if "maximum" in json_schema:
            field_params["le"] = json_schema["maximum"]
        if "exclusiveMaximum" in json_schema:
            field_params["lt"] = json_schema["exclusiveMaximum"]
        if "multipleOf" in json_schema:
            field_params["multiple_of"] = json_schema["multipleOf"]

    format_ = json_schema.get("format")
    if format_ in FORMAT_TYPE_MAP:
        pydantic_type = FORMAT_TYPE_MAP[format_]

        if format_ == "password":
            if json_schema.get("writeOnly"):
                pydantic_type = SecretBytes
        elif format_ == "uri":
            allowed_schemes = json_schema.get("scheme")
            if allowed_schemes:
                if len(allowed_schemes) == 1 and allowed_schemes[0] == "http":
                    pydantic_type = HttpUrl
                elif len(allowed_schemes) == 1 and allowed_schemes[0] == "file":
                    pydantic_type = FileUrl

        # `type_` can be a Union built from a list-form `type` (or anyOf/oneOf)
        # rather than a plain `str`, e.g. `{"type": ["string", "null"],
        # "format": "date-time"}`. Replacing the whole thing with
        # `pydantic_type` would silently drop the other members (null,
        # non-string alternatives) instead of just narrowing the string one.
        if type_ is str:
            type_ = pydantic_type
        elif get_origin(type_) is Union:
            type_ = Union[  # noqa: UP007
                tuple(
                    pydantic_type if member is str else member
                    for member in get_args(type_)
                )
            ]

    applied_str_pattern = False
    applied_str_field_params = isinstance(type_, type) and issubclass(type_, str)
    if applied_str_field_params:
        if "minLength" in json_schema:
            field_params["min_length"] = json_schema["minLength"]
        if "maxLength" in json_schema:
            field_params["max_length"] = json_schema["maxLength"]
        if "pattern" in json_schema:
            raw_pat = json_schema["pattern"]
            if isinstance(raw_pat, str) and isinstance(
                _compile_rust_pattern(raw_pat), TypeAdapter
            ):
                field_params["pattern"] = raw_pat
                applied_str_pattern = True

    # A field with an explicit default is already usable when omitted, so it
    # should not be made nullable unless the schema actually allows ``None``.
    # Only fields that fall back to a ``None`` default need the Optional wrap.
    if (
        not is_required
        and default is None
        and not (from_all_of and has_default)
        and not json_schema.get(_UNSATISFIABLE_KEY)
    ):
        type_ = Optional[type_]  # noqa: UP045 - ForwardRef does not support `|`

    if schema_extra:
        field_params["json_schema_extra"] = schema_extra

    type_ = _apply_all_of_validators(
        type_,
        json_schema,
        include_single_str_constraints=not applied_str_field_params,
        include_single_pattern=not applied_str_pattern,
        include_single_num_constraints=not applied_num_field_params,
    )

    if has_default:
        field_params["validate_default"] = True

    return type_, Field(default, **field_params)


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref to its actual schema.

    Args:
        ref: The $ref string (e.g., "#/$defs/MyType").
        root_schema: The root schema containing $defs.

    Returns:
        The resolved schema dict.
    """
    from typing import cast

    ref_path = ref.split("/")
    if ref.startswith("#/$defs/"):
        ref_schema: dict[str, Any] = root_schema["$defs"]
        start_idx = 2
    else:
        ref_schema = root_schema
        start_idx = 1
    for path in ref_path[start_idx:]:
        ref_schema = cast(dict[str, Any], ref_schema[path])
    return ref_schema


_MAX_ALL_OF_MERGE_STEPS: Final[int] = 10_000
_MAX_PATTERN_LENGTH: Final[int] = 1024
_MAX_ENUM_PATTERN_SUBJECT_LENGTH: Final[int] = 1024
_UNSATISFIABLE_KEY: Final[str] = "__crewai_unsatisfiable__"
_ALL_OF_PATTERNS_KEY: Final[str] = "__crewai_all_of_patterns__"
_ALL_OF_MULTIPLE_OF_KEY: Final[str] = "__crewai_all_of_multiple_of__"
_TYPE_CONFLICT: Final[object] = object()
_MISSING: Final[object] = object()

_ALL_OF_LOWER_BOUNDS: Final[tuple[str, ...]] = (
    "minimum",
    "exclusiveMinimum",
    "minLength",
    "minItems",
)
_ALL_OF_UPPER_BOUNDS: Final[tuple[str, ...]] = (
    "maximum",
    "exclusiveMaximum",
    "maxLength",
    "maxItems",
)
_ALL_OF_SCALAR_METADATA_KEYS: Final[tuple[str, ...]] = (
    "title",
    "description",
    "default",
    "examples",
    "format",
    "writeOnly",
    "scheme",
)
_ALL_OF_STRUCTURAL_NON_EMPTY_KEYS: Final[tuple[str, ...]] = (
    "enum",
    "const",
    "items",
    "anyOf",
    "oneOf",
    "$ref",
    "type",
    "pattern",
    _ALL_OF_PATTERNS_KEY,
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    _ALL_OF_MULTIPLE_OF_KEY,
    "minItems",
    "maxItems",
)


def _reject_unsatisfiable(_value: Any) -> Any:
    raise ValueError("Schema constraints are unsatisfiable")


def _reject_unsatisfiable_model(_cls: Any, _data: Any) -> Any:
    raise ValueError("Schema constraints are unsatisfiable")


def _compile_rust_pattern(pattern: str) -> TypeAdapter[str] | str:
    if len(pattern) > _MAX_PATTERN_LENGTH:
        return "Schema pattern exceeds maximum length"
    try:
        return TypeAdapter(
            Annotated[str, Field(pattern=pattern)],
            config=ConfigDict(regex_engine="rust-regex"),
        )
    except (SchemaError, ValueError):
        return f"Unsupported or invalid schema pattern: {pattern!r}"


def _extract_string_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, SecretBytes):
        value = value.get_secret_value()
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Bytes value is not valid UTF-8 for string constraint"
            ) from exc
    if isinstance(
        value,
        (
            Path,
            AnyUrl,
            PostgresDsn,
            MongoDsn,
            uuid.UUID,
            ipaddress.IPv4Address,
            ipaddress.IPv6Address,
            ipaddress.IPv4Interface,
            ipaddress.IPv6Interface,
            ipaddress.IPv4Network,
            ipaddress.IPv6Network,
        ),
    ):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return None


def _make_patterns_validator(
    adapters: tuple[TypeAdapter[str] | str, ...],
) -> Callable[[Any], Any]:
    def _validate(value: Any) -> Any:
        text = _extract_string_value(value)
        if text is not None:
            for adapter in adapters:
                if isinstance(adapter, str):
                    raise ValueError(adapter)
                try:
                    adapter.validate_python(text)
                except ValidationError as exc:
                    raise ValueError("String does not match required pattern") from exc
        return value

    return _validate


def _make_string_length_validator(
    min_length: int | float | None,
    max_length: int | float | None,
) -> Callable[[Any], Any]:
    def _validate(value: Any) -> Any:
        text = _extract_string_value(value)
        if text is not None:
            if min_length is not None and len(text) < min_length:
                raise ValueError(
                    f"String length {len(text)} is less than minimum {min_length}"
                )
            if max_length is not None and len(text) > max_length:
                raise ValueError(
                    f"String length {len(text)} exceeds maximum {max_length}"
                )
        return value

    return _validate


def _make_numeric_bounds_validator(
    minimum: int | float | None,
    exclusive_minimum: int | float | None,
    maximum: int | float | None,
    exclusive_maximum: int | float | None,
) -> Callable[[Any], Any]:
    def _validate(value: Any) -> Any:
        if _is_json_number(value):
            if minimum is not None and value < minimum:
                raise ValueError(f"Value {value} is less than minimum {minimum}")
            if exclusive_minimum is not None and value <= exclusive_minimum:
                raise ValueError(
                    f"Value {value} must be greater than {exclusive_minimum}"
                )
            if maximum is not None and value > maximum:
                raise ValueError(f"Value {value} exceeds maximum {maximum}")
            if exclusive_maximum is not None and value >= exclusive_maximum:
                raise ValueError(f"Value {value} must be less than {exclusive_maximum}")
        return value

    return _validate


def _make_array_length_validator(
    min_items: int | float | None,
    max_items: int | float | None,
) -> Callable[[Any], Any]:
    def _validate(value: Any) -> Any:
        if isinstance(value, list):
            if min_items is not None and len(value) < min_items:
                raise ValueError(
                    f"Array length {len(value)} is less than minimum {min_items}"
                )
            if max_items is not None and len(value) > max_items:
                raise ValueError(
                    f"Array length {len(value)} exceeds maximum {max_items}"
                )
        return value

    return _validate


def _make_multiple_of_validator(
    divisors: tuple[int | float, ...],
) -> Callable[[Any], Any]:
    def _validate(value: Any) -> Any:
        if _is_json_number(value):
            for divisor in divisors:
                if divisor <= 0:
                    raise ValueError("multipleOf divisor must be positive")
                if isinstance(value, int) and isinstance(divisor, int):
                    if value % divisor != 0:
                        raise ValueError(
                            f"Value {value} is not a multiple of {divisor}"
                        )
                else:
                    quotient = value / divisor
                    if abs(quotient - round(quotient)) > 1e-9:
                        raise ValueError(
                            f"Value {value} is not a multiple of {divisor}"
                        )
        return value

    return _validate


def _apply_all_of_validators(
    type_: Any,
    schema: dict[str, Any],
    *,
    include_single_str_constraints: bool = True,
    include_single_pattern: bool | None = None,
    include_single_num_constraints: bool = True,
) -> Any:
    if schema.get(_UNSATISFIABLE_KEY):
        return Annotated[Any, AfterValidator(_reject_unsatisfiable)]

    check_single_pattern = (
        include_single_str_constraints
        if include_single_pattern is None
        else include_single_pattern
    )
    raw_patterns = schema.get(_ALL_OF_PATTERNS_KEY)
    pattern_list: tuple[str, ...] = ()
    if isinstance(raw_patterns, (tuple, list)):
        pattern_list = tuple(p for p in raw_patterns if isinstance(p, str))
    elif check_single_pattern:
        single_pat = schema.get("pattern")
        if isinstance(single_pat, str):
            pattern_list = (single_pat,)
    if pattern_list:
        adapters = tuple(_compile_rust_pattern(pat) for pat in pattern_list)
        type_ = Annotated[type_, BeforeValidator(_make_patterns_validator(adapters))]

    if include_single_str_constraints:
        raw_min_len = schema.get("minLength")
        raw_max_len = schema.get("maxLength")
        min_length = raw_min_len if _is_json_number(raw_min_len) else None
        max_length = raw_max_len if _is_json_number(raw_max_len) else None
        if min_length is not None or max_length is not None:
            type_ = Annotated[
                type_,
                BeforeValidator(_make_string_length_validator(min_length, max_length)),
            ]

    if include_single_num_constraints:
        raw_min = schema.get("minimum")
        raw_excl_min = schema.get("exclusiveMinimum")
        raw_max = schema.get("maximum")
        raw_excl_max = schema.get("exclusiveMaximum")
        minimum = raw_min if _is_json_number(raw_min) else None
        excl_min = raw_excl_min if _is_json_number(raw_excl_min) else None
        maximum = raw_max if _is_json_number(raw_max) else None
        excl_max = raw_excl_max if _is_json_number(raw_excl_max) else None
        if any(b is not None for b in (minimum, excl_min, maximum, excl_max)):
            type_ = Annotated[
                type_,
                AfterValidator(
                    _make_numeric_bounds_validator(minimum, excl_min, maximum, excl_max)
                ),
            ]

    raw_divisors = schema.get(_ALL_OF_MULTIPLE_OF_KEY)
    divisors: tuple[int | float, ...] = ()
    if isinstance(raw_divisors, (tuple, list)):
        divisors = tuple(d for d in raw_divisors if _is_json_number(d))
    elif include_single_num_constraints:
        single_div = schema.get("multipleOf")
        if _is_json_number(single_div):
            divisors = (single_div,)
    if divisors:
        type_ = Annotated[type_, AfterValidator(_make_multiple_of_validator(divisors))]

    raw_min_items = schema.get("minItems")
    raw_max_items = schema.get("maxItems")
    min_items = raw_min_items if _is_json_number(raw_min_items) else None
    max_items = raw_max_items if _is_json_number(raw_max_items) else None
    if min_items is not None or max_items is not None:
        type_ = Annotated[
            type_,
            AfterValidator(_make_array_length_validator(min_items, max_items)),
        ]

    return type_


def _is_json_number(val: Any) -> TypeGuard[int | float]:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _json_values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_json_number(a) and _is_json_number(b):
        return bool(a == b)
    if type(a) is not type(b):
        return False
    return bool(a == b)


def _value_matches_json_type(value: Any, json_type: Any) -> bool:
    if isinstance(json_type, list):
        return any(_value_matches_json_type(value, member) for member in json_type)
    if json_type == "null":
        return value is None
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return _is_json_number(value)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return True


def _value_matches_constraints(
    value: Any,
    merged: dict[str, Any],
    pattern_adapters: tuple[TypeAdapter[str] | str, ...],
    multiple_ofs: list[int | float],
    budget: list[int],
) -> bool:
    if "type" in merged and not _value_matches_json_type(value, merged["type"]):
        return False
    if isinstance(value, str):
        if pattern_adapters:
            if len(value) > _MAX_ENUM_PATTERN_SUBJECT_LENGTH:
                return False
            for adapter in pattern_adapters:
                if budget[0] <= 0:
                    raise ValueError("Schema allOf complexity limit exceeded")
                budget[0] -= 1
                if isinstance(adapter, str):
                    return False
                try:
                    adapter.validate_python(value)
                except ValidationError:
                    return False
        min_len = merged.get("minLength")
        if _is_json_number(min_len) and len(value) < min_len:
            return False
        max_len = merged.get("maxLength")
        if _is_json_number(max_len) and len(value) > max_len:
            return False
    if isinstance(value, list):
        min_items = merged.get("minItems")
        if _is_json_number(min_items) and len(value) < min_items:
            return False
        max_items = merged.get("maxItems")
        if _is_json_number(max_items) and len(value) > max_items:
            return False
    if _is_json_number(value):
        minimum = merged.get("minimum")
        if _is_json_number(minimum) and value < minimum:
            return False
        excl_min = merged.get("exclusiveMinimum")
        if _is_json_number(excl_min) and value <= excl_min:
            return False
        maximum = merged.get("maximum")
        if _is_json_number(maximum) and value > maximum:
            return False
        excl_max = merged.get("exclusiveMaximum")
        if _is_json_number(excl_max) and value >= excl_max:
            return False
        for divisor in multiple_ofs:
            if divisor <= 0:
                return False
            if isinstance(value, int) and isinstance(divisor, int):
                if value % divisor != 0:
                    return False
            else:
                quotient = value / divisor
                if abs(quotient - round(quotient)) > 1e-9:
                    return False
    return True


def _merge_json_schema_types(existing: Any, new: Any) -> Any:
    """Narrow two JSON Schema ``type`` declarations for ``allOf``."""
    if existing is None:
        return new
    if new is None:
        return existing
    if isinstance(existing, str) and isinstance(new, str):
        if existing == new:
            return existing
        if {existing, new} == {"integer", "number"}:
            return "integer"
        return _TYPE_CONFLICT
    existing_list = existing if isinstance(existing, list) else [existing]
    new_list = new if isinstance(new, list) else [new]
    narrowed: list[Any] = []
    for member in existing_list:
        if member in new_list or (member == "integer" and "number" in new_list):
            if member not in narrowed:
                narrowed.append(member)
        elif member == "number" and "integer" in new_list:
            if "integer" not in narrowed:
                narrowed.append("integer")
    if len(narrowed) == 1:
        return narrowed[0]
    if len(narrowed) > 1:
        return narrowed
    return _TYPE_CONFLICT


def _merge_all_of_schemas(
    schemas: list[dict[str, Any]],
    root_schema: dict[str, Any],
    *,
    outer_schema: dict[str, Any] | None = None,
    _seen: set[int] | None = None,
    _budget: list[int] | None = None,
    _memo: dict[tuple[int, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge multiple ``allOf`` schemas conjunctively with complexity bounds.

    Intersects overlapping properties and nested ``items``, ANDs multiple
    ``pattern`` and ``multipleOf`` constraints, intersects ``enum`` / ``const`` /
    ``type`` / ``anyOf`` / ``oneOf`` (failing closed on unsatisfiable
    combinations), flattens nested ``allOf`` chains, and bounds total work with
    shared memoization across DAG branches.
    """
    merged: dict[str, Any] = {}
    seen: set[int] = set() if _seen is None else _seen
    budget: list[int] = [_MAX_ALL_OF_MERGE_STEPS] if _budget is None else _budget
    memo: dict[tuple[int, ...], dict[str, Any]] = {} if _memo is None else _memo

    unsatisfiable = False
    prop_branches: dict[str, list[dict[str, Any]]] = {}
    prop_branch_ids: dict[str, set[int]] = {}
    prop_unsat: set[str] = set()
    outer_props: dict[str, dict[str, Any]] = {}
    item_false = False
    item_dicts: list[dict[str, Any]] = []
    item_dict_ids: set[int] = set()
    item_other: Any = _MISSING
    enum_vals: list[Any] | None = None
    has_const = False
    const_val: Any = None
    patterns: list[str] = []
    multiple_ofs: list[int | float] = []
    union_groups: list[list[Any]] = []

    def _apply(sub: Any, *, is_outer: bool = False) -> None:
        nonlocal unsatisfiable, item_false, item_other
        nonlocal enum_vals, has_const, const_val

        if sub is False:
            unsatisfiable = True
            return
        if not isinstance(sub, dict):
            return
        if budget[0] <= 0:
            raise ValueError("Schema allOf complexity limit exceeded")
        budget[0] -= 1

        if sub.get(_UNSATISFIABLE_KEY):
            unsatisfiable = True
            return

        if not is_outer:
            sub_id = id(sub)
            if sub_id in seen:
                return
            seen.add(sub_id)

        ref = sub.get("$ref")
        if isinstance(ref, str):
            _apply(_resolve_ref(ref, root_schema), is_outer=False)

        nested_all_of = sub.get("allOf")
        if not is_outer and isinstance(nested_all_of, list):
            for nested in nested_all_of:
                _apply(nested, is_outer=False)

        if "type" in sub:
            narrowed_type = _merge_json_schema_types(merged.get("type"), sub["type"])
            if narrowed_type is _TYPE_CONFLICT:
                unsatisfiable = True
            else:
                merged["type"] = narrowed_type

        sub_props = sub.get("properties")
        if isinstance(sub_props, dict) and sub_props:
            for prop_name, prop_schema in sub_props.items():
                if prop_schema is False:
                    prop_unsat.add(prop_name)
                elif isinstance(prop_schema, dict):
                    if is_outer:
                        outer_props[prop_name] = prop_schema
                    else:
                        b_ids = prop_branch_ids.setdefault(prop_name, set())
                        if id(prop_schema) not in b_ids:
                            b_ids.add(id(prop_schema))
                            prop_branches.setdefault(prop_name, []).append(prop_schema)

        sub_required = sub.get("required")
        if isinstance(sub_required, list) and sub_required:
            req_list: list[Any] = merged.setdefault("required", [])
            for field in sub_required:
                if field not in req_list:
                    req_list.append(field)

        if "enum" in sub:
            sub_enum = sub["enum"]
            if isinstance(sub_enum, list):
                if not sub_enum:
                    unsatisfiable = True
                elif enum_vals is None:
                    enum_vals = list(sub_enum)
                else:
                    enum_vals = [
                        v
                        for v in enum_vals
                        if any(
                            _json_values_equal(v, candidate) for candidate in sub_enum
                        )
                    ]
                    if not enum_vals:
                        unsatisfiable = True

        if "const" in sub:
            cand_const = sub["const"]
            if not has_const:
                has_const = True
                const_val = cand_const
            elif not _json_values_equal(const_val, cand_const):
                unsatisfiable = True

        if "items" in sub:
            sub_items = sub["items"]
            if sub_items is False:
                item_false = True
            elif isinstance(sub_items, dict):
                if id(sub_items) not in item_dict_ids:
                    item_dict_ids.add(id(sub_items))
                    item_dicts.append(sub_items)
            elif item_other is _MISSING:
                item_other = sub_items

        internal_patterns = sub.get(_ALL_OF_PATTERNS_KEY)
        if isinstance(internal_patterns, (tuple, list)):
            for pat in internal_patterns:
                if isinstance(pat, str) and pat not in patterns:
                    patterns.append(pat)
        else:
            raw_pat = sub.get("pattern")
            if isinstance(raw_pat, str) and raw_pat not in patterns:
                patterns.append(raw_pat)

        internal_multiples = sub.get(_ALL_OF_MULTIPLE_OF_KEY)
        if isinstance(internal_multiples, (tuple, list)):
            for div in internal_multiples:
                if _is_json_number(div) and div not in multiple_ofs:
                    multiple_ofs.append(div)
        else:
            raw_mult = sub.get("multipleOf")
            if _is_json_number(raw_mult) and raw_mult not in multiple_ofs:
                multiple_ofs.append(raw_mult)

        for key in _ALL_OF_LOWER_BOUNDS:
            if key in sub:
                val = sub[key]
                cur = merged.get(key)
                if _is_json_number(val) and _is_json_number(cur):
                    merged[key] = max(cur, val)
                elif is_outer or key not in merged:
                    merged[key] = val

        for key in _ALL_OF_UPPER_BOUNDS:
            if key in sub:
                val = sub[key]
                cur = merged.get(key)
                if _is_json_number(val) and _is_json_number(cur):
                    merged[key] = min(cur, val)
                elif is_outer or key not in merged:
                    merged[key] = val

        for union_key in ("anyOf", "oneOf"):
            if union_key in sub:
                u_list = sub[union_key]
                if isinstance(u_list, list):
                    if not u_list:
                        unsatisfiable = True
                    else:
                        union_groups.append(u_list)

        for key in _ALL_OF_SCALAR_METADATA_KEYS:
            if key in sub:
                if is_outer:
                    merged[key] = sub[key]
                else:
                    merged.setdefault(key, sub[key])

    for schema in schemas:
        _apply(schema, is_outer=False)

    if outer_schema is not None:
        _apply(outer_schema, is_outer=True)

    # Check inverted bounds after merging all branches
    for low_key, high_key, strict in (
        ("minLength", "maxLength", False),
        ("minItems", "maxItems", False),
        ("minimum", "maximum", False),
        ("exclusiveMinimum", "maximum", True),
        ("minimum", "exclusiveMaximum", True),
        ("exclusiveMinimum", "exclusiveMaximum", True),
    ):
        low_val = merged.get(low_key)
        high_val = merged.get(high_key)
        if _is_json_number(low_val) and _is_json_number(high_val):
            if (low_val >= high_val) if strict else (low_val > high_val):
                unsatisfiable = True

    if unsatisfiable:
        return {_UNSATISFIABLE_KEY: True}

    if patterns:
        first_compiled = _compile_rust_pattern(patterns[0])
        if isinstance(first_compiled, TypeAdapter):
            merged["pattern"] = patterns[0]
        if len(patterns) > 1 or not isinstance(first_compiled, TypeAdapter):
            merged[_ALL_OF_PATTERNS_KEY] = tuple(patterns)

    if multiple_ofs:
        merged["multipleOf"] = multiple_ofs[0]
        if len(multiple_ofs) > 1:
            merged[_ALL_OF_MULTIPLE_OF_KEY] = tuple(multiple_ofs)

    if prop_branches or outer_props or prop_unsat:
        merged_props: dict[str, Any] = {}
        all_prop_names = list(
            dict.fromkeys([*prop_branches.keys(), *outer_props.keys(), *prop_unsat])
        )
        for prop_name in all_prop_names:
            if prop_name in prop_unsat:
                merged_props[prop_name] = {_UNSATISFIABLE_KEY: True}
                continue
            b_list = prop_branches.get(prop_name, [])
            o_prop = outer_props.get(prop_name)
            if o_prop is None:
                merged_props[prop_name] = (
                    b_list[0] if len(b_list) == 1 else {"allOf": b_list}
                )
            elif not b_list or (len(b_list) == 1 and b_list[0] is o_prop):
                merged_props[prop_name] = o_prop
            else:
                o_all_of = o_prop.get("allOf")
                o_rest = {pk: pv for pk, pv in o_prop.items() if pk != "allOf"}
                combined_all_of = [*b_list]
                if isinstance(o_all_of, list):
                    for nested_branch in o_all_of:
                        if nested_branch not in combined_all_of:
                            combined_all_of.append(nested_branch)
                merged_props[prop_name] = {**o_rest, "allOf": combined_all_of}
        merged["properties"] = merged_props

    if item_false:
        merged["items"] = False
    elif len(item_dicts) == 1:
        merged["items"] = item_dicts[0]
    elif len(item_dicts) > 1:
        memo_key = tuple(id(d) for d in item_dicts)
        if memo_key in memo:
            merged["items"] = memo[memo_key]
        else:
            memo[memo_key] = {}
            merged_items = _merge_all_of_schemas(
                item_dicts,
                root_schema,
                _seen=set(seen),
                _budget=budget,
                _memo=memo,
            )
            memo[memo_key] = merged_items
            merged["items"] = merged_items
    elif item_other is not _MISSING:
        merged["items"] = item_other

    if has_const or enum_vals is not None:
        pattern_adapters = (
            tuple(_compile_rust_pattern(pat) for pat in patterns) if patterns else ()
        )
        if has_const:
            if enum_vals is not None and not any(
                _json_values_equal(const_val, e) for e in enum_vals
            ):
                return {_UNSATISFIABLE_KEY: True}
            if not _value_matches_constraints(
                const_val, merged, pattern_adapters, multiple_ofs, budget
            ):
                return {_UNSATISFIABLE_KEY: True}
            merged["const"] = const_val
        elif enum_vals is not None:
            filtered = [
                v
                for v in enum_vals
                if _value_matches_constraints(
                    v, merged, pattern_adapters, multiple_ofs, budget
                )
            ]
            if not filtered:
                return {_UNSATISFIABLE_KEY: True}
            merged["enum"] = filtered

    if union_groups:
        structural_keys = (
            "type",
            "enum",
            "const",
            "pattern",
            _ALL_OF_PATTERNS_KEY,
            "multipleOf",
            _ALL_OF_MULTIPLE_OF_KEY,
            "properties",
            "required",
            "items",
            *_ALL_OF_LOWER_BOUNDS,
            *_ALL_OF_UPPER_BOUNDS,
        )
        has_base_structural = any(k in merged for k in structural_keys)
        if len(union_groups) == 1 and not has_base_structural:
            merged["anyOf"] = union_groups[0]
        else:
            meta = {k: merged[k] for k in _ALL_OF_SCALAR_METADATA_KEYS if k in merged}
            base_only = {
                k: v for k, v in merged.items() if k not in _ALL_OF_SCALAR_METADATA_KEYS
            }
            candidates: list[dict[str, Any]] = (
                [base_only] if has_base_structural else [{}]
            )
            for group in union_groups:
                if len(candidates) * len(group) > budget[0]:
                    raise ValueError("Schema allOf complexity limit exceeded")
                next_candidates: list[dict[str, Any]] = []
                for cand in candidates:
                    for alt in group:
                        if alt is False:
                            continue
                        alt_dict = {} if alt is True else alt
                        if not isinstance(alt_dict, dict):
                            continue
                        pair = _merge_all_of_schemas(
                            [cand, alt_dict],
                            root_schema,
                            _seen=set(seen),
                            _budget=budget,
                            _memo=memo,
                        )
                        if not pair.get(_UNSATISFIABLE_KEY):
                            next_candidates.append(pair)
                candidates = next_candidates
                if not candidates:
                    return {_UNSATISFIABLE_KEY: True}
            if len(candidates) == 1:
                result = {**candidates[0], **meta}
                if "type" not in result and result.get("properties"):
                    result["type"] = "object"
                return result
            return {**meta, "anyOf": candidates}

    if "type" not in merged and merged.get("properties"):
        merged["type"] = "object"

    return merged


def _lower_union_arm(
    arm: Any,
    root_schema: dict[str, Any],
    *,
    name_: str,
    enrich_descriptions: bool,
    in_progress: dict[int, Any] | None,
) -> Any:
    if isinstance(arm, dict):
        return _json_schema_to_pydantic_type(
            arm,
            root_schema,
            name_=name_,
            enrich_descriptions=enrich_descriptions,
            in_progress=in_progress,
            enforce_constraints=True,
        )
    if arm is True:
        return Any
    if arm is False:
        return Annotated[Any, AfterValidator(_reject_unsatisfiable)]
    raise ValueError(f"Unsupported schema in union arm: {arm!r}")


def _json_schema_to_pydantic_type(
    json_schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    name_: str | None = None,
    enrich_descriptions: bool = False,
    in_progress: dict[int, Any] | None = None,
    enforce_constraints: bool = False,
) -> Any:
    """Convert a JSON schema to a Python/Pydantic type.

    Args:
        json_schema: The JSON schema to convert.
        root_schema: The root schema for resolving $ref.
        name_: Optional name for nested models.
        enrich_descriptions: Propagated to nested model creation.
        in_progress: Map of ``id(schema_dict)`` to the Pydantic model
            currently being built for that schema, or to a placeholder name
            as a plain ``str`` while the model is still being constructed.
            Populated by :func:`_build_model_from_schema`. Enables cycle
            detection so a self-referential ``$ref`` graph resolves to a
            :class:`ForwardRef` back-edge rather than recursing forever.
        enforce_constraints: When True, attach surviving scalar/array constraint
            validators directly to nested leaf/union types (e.g., inside ``items``
            or ``anyOf``/``oneOf`` arms) that do not pass through
            :func:`_json_schema_to_pydantic_field`.

    Returns:
        A Python type corresponding to the JSON schema.
    """
    if in_progress is not None:
        cached = in_progress.get(id(json_schema))
        if isinstance(cached, str):
            return ForwardRef(cached)
        if cached is not None:
            return cached

    if json_schema.get(_UNSATISFIABLE_KEY):
        return Annotated[Any, AfterValidator(_reject_unsatisfiable)]

    all_of_schemas = json_schema.get("allOf")
    if all_of_schemas:
        has_structural_siblings = bool(
            json_schema.get("properties")
            or json_schema.get("required")
            or any(key in json_schema for key in _ALL_OF_STRUCTURAL_NON_EMPTY_KEYS)
        )
        if len(all_of_schemas) == 1 and not has_structural_siblings:
            return _json_schema_to_pydantic_type(
                all_of_schemas[0],
                root_schema,
                name_=name_,
                enrich_descriptions=enrich_descriptions,
                in_progress=in_progress,
                enforce_constraints=enforce_constraints,
            )
        merged = _merge_all_of_schemas(
            all_of_schemas,
            root_schema,
            outer_schema=json_schema,
        )
        if merged.get(_UNSATISFIABLE_KEY):
            return Annotated[Any, AfterValidator(_reject_unsatisfiable)]
        if merged.get("type") == "object" and merged.get("properties"):
            if in_progress is not None:
                return _build_model_from_schema(
                    json_schema,
                    root_schema,
                    model_name=name_,
                    enrich_descriptions=enrich_descriptions,
                    in_progress=in_progress,
                )
        return _json_schema_to_pydantic_type(
            merged,
            root_schema,
            name_=name_,
            enrich_descriptions=enrich_descriptions,
            in_progress=in_progress,
            enforce_constraints=enforce_constraints,
        )

    ref = json_schema.get("$ref")
    if ref:
        ref_schema = _resolve_ref(ref, root_schema)
        return _json_schema_to_pydantic_type(
            ref_schema,
            root_schema,
            name_=name_,
            enrich_descriptions=enrich_descriptions,
            in_progress=in_progress,
            enforce_constraints=enforce_constraints,
        )

    enum_values = json_schema.get("enum")
    if enum_values:
        lit_type: Any = Literal[tuple(enum_values)]
        return (
            _apply_all_of_validators(lit_type, json_schema)
            if enforce_constraints
            else lit_type
        )

    if "const" in json_schema:
        const_type: Any = Literal[json_schema["const"]]
        return (
            _apply_all_of_validators(const_type, json_schema)
            if enforce_constraints
            else const_type
        )

    any_of_schemas = []
    if "anyOf" in json_schema or "oneOf" in json_schema:
        any_of_schemas = json_schema.get("anyOf", []) + json_schema.get("oneOf", [])
    if any_of_schemas:
        any_of_types = [
            _lower_union_arm(
                schema,
                root_schema,
                name_=f"{name_ or 'Union'}Option{i}",
                enrich_descriptions=enrich_descriptions,
                in_progress=in_progress,
            )
            for i, schema in enumerate(any_of_schemas)
        ]
        union_type: Any = Union[tuple(any_of_types)]  # noqa: UP007
        return (
            _apply_all_of_validators(union_type, json_schema)
            if enforce_constraints
            else union_type
        )

    type_ = json_schema.get("type")

    if isinstance(type_, list):
        # JSON Schema also allows "type" to be an array, e.g.
        # {"type": ["string", "null"]} -- the .NET/System.Text.Json-style
        # way of expressing a nullable field. Pydantic's own schema
        # generation instead uses anyOf/oneOf for this (handled above), so
        # external tool schemas (e.g. from a non-Python MCP server) are the
        # main source of this form. Treat each entry the same way anyOf's
        # members are handled just above: build a Union of the
        # corresponding Python types. A single-element list collapses to
        # that one type, matching typing.Union's own behavior.
        member_types = [
            _json_schema_to_pydantic_type(
                {**json_schema, "type": member},
                root_schema,
                name_=f"{name_ or 'Union'}Option{i}",
                enrich_descriptions=enrich_descriptions,
                in_progress=in_progress,
                enforce_constraints=enforce_constraints,
            )
            for i, member in enumerate(type_)
        ]
        return Union[tuple(member_types)]  # noqa: UP007

    if type_ == "string":
        return (
            _apply_all_of_validators(str, json_schema) if enforce_constraints else str
        )
    if type_ == "integer":
        return (
            _apply_all_of_validators(int, json_schema) if enforce_constraints else int
        )
    if type_ == "number":
        return (
            _apply_all_of_validators(float, json_schema)
            if enforce_constraints
            else float
        )
    if type_ == "boolean":
        return bool
    if type_ == "array":
        items_schema = json_schema.get("items")
        array_type: Any
        if items_schema is False:
            array_type = list[Annotated[Any, AfterValidator(_reject_unsatisfiable)]]
        elif isinstance(items_schema, dict):
            item_type = _json_schema_to_pydantic_type(
                items_schema,
                root_schema,
                name_=name_,
                enrich_descriptions=enrich_descriptions,
                in_progress=in_progress,
                enforce_constraints=True,
            )
            array_type = list[item_type]  # type: ignore[valid-type]
        elif items_schema:
            item_type = _json_schema_to_pydantic_type(
                items_schema,
                root_schema,
                name_=name_,
                enrich_descriptions=enrich_descriptions,
                in_progress=in_progress,
                enforce_constraints=True,
            )
            array_type = list[item_type]  # type: ignore[valid-type]
        else:
            array_type = list
        return (
            _apply_all_of_validators(array_type, json_schema)
            if enforce_constraints
            else array_type
        )
    if type_ == "object":
        properties = json_schema.get("properties")
        if properties:
            if in_progress is not None:
                return _build_model_from_schema(
                    json_schema,
                    root_schema,
                    model_name=name_,
                    enrich_descriptions=enrich_descriptions,
                    in_progress=in_progress,
                )
            json_schema_ = json_schema.copy()
            if json_schema_.get("title") is None:
                json_schema_["title"] = name_ or "DynamicModel"
            return create_model_from_schema(
                json_schema_,
                root_schema=root_schema,
                enrich_descriptions=enrich_descriptions,
            )
        return dict
    if type_ == "null":
        return None
    if type_ is None:
        return (
            _apply_all_of_validators(Any, json_schema) if enforce_constraints else Any
        )
    raise ValueError(f"Unsupported JSON schema type: {type_} from {json_schema}")

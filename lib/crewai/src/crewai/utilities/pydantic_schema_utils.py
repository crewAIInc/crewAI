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
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union
import uuid

from pydantic import (
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    AnyUrl,
    BaseModel,
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
    create_model as create_model_base,
)
from pydantic.networks import (  # type: ignore[attr-defined]
    IPv4Address,
    IPv6Address,
    IPvAnyAddress,
    IPvAnyInterface,
    IPvAnyNetwork,
)


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

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                def_name = ref.replace("#/$defs/", "")
                if def_name in defs:
                    return _resolve(deepcopy(defs[def_name]))
                raise KeyError(f"Definition '{def_name}' not found in $defs.")
            return {k: _resolve(v) for k, v in node.items()}

        if isinstance(node, list):
            return [_resolve(i) for i in node]

        return node

    return _resolve(schema_copy)  # type: ignore[no-any-return]


def add_key_in_dict_recursively(
    d: dict[str, Any], key: str, value: Any, criteria: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    """Recursively adds a key/value pair to all nested dicts matching `criteria`.

    Args:
        d: The dictionary to modify.
        key: The key to add.
        value: The value to add.
        criteria: A function that returns True for dicts that should receive the key.

    Returns:
        The modified dictionary.
    """
    if isinstance(d, dict):
        if criteria(d) and key not in d:
            d[key] = value
        for v in d.values():
            add_key_in_dict_recursively(v, key, value, criteria)
    elif isinstance(d, list):
        for i in d:
            add_key_in_dict_recursively(i, key, value, criteria)
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


def convert_oneof_to_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert oneOf to anyOf for OpenAI compatibility.

    OpenAI's Structured Outputs support anyOf better than oneOf.
    This recursively converts all oneOf occurrences to anyOf.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Modified schema with anyOf instead of oneOf.
    """
    if isinstance(schema, dict):
        if "oneOf" in schema:
            schema["anyOf"] = schema.pop("oneOf")

        for value in schema.values():
            if isinstance(value, dict):
                convert_oneof_to_anyof(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        convert_oneof_to_anyof(item)

    return schema


def ensure_all_properties_required(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure all properties are in the required array for OpenAI strict mode.

    OpenAI's strict structured outputs require all properties to be listed
    in the required array. This recursively updates all objects to include
    all their properties in required.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Modified schema with all properties marked as required.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema["properties"]
            if properties:
                schema["required"] = list(properties.keys())

        for value in schema.values():
            if isinstance(value, dict):
                ensure_all_properties_required(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        ensure_all_properties_required(item)

    return schema


def generate_model_description(model: type[BaseModel]) -> dict[str, Any]:
    """Generate JSON schema description of a Pydantic model.

    This function takes a Pydantic model class and returns its JSON schema,
    which includes full type information, discriminators, and all metadata.
    The schema is dereferenced to inline all $ref references for better LLM understanding.

    Args:
        model: A Pydantic model class.

    Returns:
        A JSON schema dictionary representation of the model.
    """
    json_schema = model.model_json_schema(ref_template="#/$defs/{model}")

    json_schema = add_key_in_dict_recursively(
        json_schema,
        key="additionalProperties",
        value=False,
        criteria=lambda d: d.get("type") == "object"
        and "additionalProperties" not in d,
    )

    json_schema = resolve_refs(json_schema)

    json_schema.pop("$defs", None)
    json_schema = fix_discriminator_mappings(json_schema)
    json_schema = convert_oneof_to_anyof(json_schema)
    json_schema = ensure_all_properties_required(json_schema)

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


def create_model_from_schema(  # type: ignore[no-any-unimported]
    json_schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
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
    effective_root = root_schema or json_schema

    if "allOf" in json_schema:
        json_schema = _merge_all_of_schemas(json_schema["allOf"], effective_root)
        if "title" not in json_schema and "title" in (root_schema or {}):
            json_schema["title"] = (root_schema or {}).get("title")

    model_name = json_schema.get("title", "DynamicModel")
    field_definitions = {
        name: _json_schema_to_pydantic_field(
            name, prop, json_schema.get("required", []), effective_root
        )
        for name, prop in (json_schema.get("properties", {}) or {}).items()
    }

    return create_model_base(
        model_name,
        __config__=__config__,
        __base__=__base__,
        __module__=__module__,
        __validators__=__validators__,
        __cls_kwargs__=__cls_kwargs__,
        **field_definitions,
    )


def _json_schema_to_pydantic_field(
    name: str,
    json_schema: dict[str, Any],
    required: list[str],
    root_schema: dict[str, Any],
) -> Any:
    """Convert a JSON schema property to a Pydantic field definition.

    Args:
        name: The field name.
        json_schema: The JSON schema for this field.
        required: List of required field names.
        root_schema: The root schema for resolving $ref.

    Returns:
        A tuple of (type, Field) for use with create_model.
    """
    type_ = _json_schema_to_pydantic_type(json_schema, root_schema, name_=name.title())
    description = json_schema.get("description")
    examples = json_schema.get("examples")
    is_required = name in required

    field_params: dict[str, Any] = {}
    schema_extra: dict[str, Any] = {}

    if description:
        field_params["description"] = description
    if examples:
        schema_extra["examples"] = examples

    default = ... if is_required else None

    if isinstance(type_, type) and issubclass(type_, (int, float)):
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

        type_ = pydantic_type

    if isinstance(type_, type) and issubclass(type_, str):
        if "minLength" in json_schema:
            field_params["min_length"] = json_schema["minLength"]
        if "maxLength" in json_schema:
            field_params["max_length"] = json_schema["maxLength"]
        if "pattern" in json_schema:
            field_params["pattern"] = json_schema["pattern"]

    if not is_required:
        type_ = type_ | None

    if schema_extra:
        field_params["json_schema_extra"] = schema_extra

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


def _merge_all_of_schemas(
    schemas: list[dict[str, Any]],
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    """Merge multiple allOf schemas into a single schema.

    Combines properties and required fields from all schemas.

    Args:
        schemas: List of schemas to merge.
        root_schema: The root schema for resolving $ref.

    Returns:
        Merged schema with combined properties and required fields.
    """
    merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for schema in schemas:
        if "$ref" in schema:
            schema = _resolve_ref(schema["$ref"], root_schema)

        if "properties" in schema:
            merged["properties"].update(schema["properties"])

        if "required" in schema:
            for field in schema["required"]:
                if field not in merged["required"]:
                    merged["required"].append(field)

        if "title" in schema and "title" not in merged:
            merged["title"] = schema["title"]

    return merged


def _json_schema_to_pydantic_type(
    json_schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    name_: str | None = None,
) -> Any:
    """Convert a JSON schema to a Python/Pydantic type.

    Args:
        json_schema: The JSON schema to convert.
        root_schema: The root schema for resolving $ref.
        name_: Optional name for nested models.

    Returns:
        A Python type corresponding to the JSON schema.
    """
    ref = json_schema.get("$ref")
    if ref:
        ref_schema = _resolve_ref(ref, root_schema)
        return _json_schema_to_pydantic_type(ref_schema, root_schema, name_=name_)

    enum_values = json_schema.get("enum")
    if enum_values:
        return Literal[tuple(enum_values)]

    if "const" in json_schema:
        return Literal[json_schema["const"]]

    any_of_schemas = []
    if "anyOf" in json_schema or "oneOf" in json_schema:
        any_of_schemas = json_schema.get("anyOf", []) + json_schema.get("oneOf", [])
    if any_of_schemas:
        any_of_types = [
            _json_schema_to_pydantic_type(schema, root_schema)
            for schema in any_of_schemas
        ]
        return Union[tuple(any_of_types)]  # noqa: UP007

    all_of_schemas = json_schema.get("allOf")
    if all_of_schemas:
        if len(all_of_schemas) == 1:
            return _json_schema_to_pydantic_type(
                all_of_schemas[0], root_schema, name_=name_
            )
        merged = _merge_all_of_schemas(all_of_schemas, root_schema)
        return _json_schema_to_pydantic_type(merged, root_schema, name_=name_)

    type_ = json_schema.get("type")

    if type_ == "string":
        return str
    if type_ == "integer":
        return int
    if type_ == "number":
        return float
    if type_ == "boolean":
        return bool
    if type_ == "array":
        items_schema = json_schema.get("items")
        if items_schema:
            item_type = _json_schema_to_pydantic_type(
                items_schema, root_schema, name_=name_
            )
            return list[item_type]  # type: ignore[valid-type]
        return list
    if type_ == "object":
        properties = json_schema.get("properties")
        if properties:
            json_schema_ = json_schema.copy()
            if json_schema_.get("title") is None:
                json_schema_["title"] = name_
            return create_model_from_schema(json_schema_, root_schema=root_schema)
        return dict
    if type_ == "null":
        return None
    if type_ is None:
        return Any
    raise ValueError(f"Unsupported JSON schema type: {type_} from {json_schema}")

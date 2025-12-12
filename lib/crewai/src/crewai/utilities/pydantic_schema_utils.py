"""Utilities for generating JSON schemas from Pydantic models.

This module provides functions for converting Pydantic models to JSON schemas
suitable for use with LLMs and tool definitions.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pydantic import BaseModel


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

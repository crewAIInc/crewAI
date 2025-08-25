"""Pydantic schema utilities for RAG configuration."""

from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin, Literal
from types import UnionType
from pydantic_core import CoreSchema, core_schema
from pydantic import GetCoreSchemaHandler
from pydantic_settings import BaseSettings


def serialize_pydantic_settings(settings: BaseSettings | None) -> dict[str, Any] | None:
    """Serialize Pydantic settings object to dict.

    Args:
        settings: Pydantic settings object with 'dict' (v1) or 'model_dump' (v2) method.

    Returns:
        Dictionary representation of settings, None if input is None,
        or unchanged object if no serialization methods exist.
    """
    if settings is None:
        return None
    if hasattr(settings, "model_dump"):
        return settings.model_dump()
    if hasattr(settings, "dict"):
        return settings.dict()  # type: ignore[deprecated]
    raise ValueError("Settings object lacks 'dict' or 'model_dump' method.")


def create_dataclass_schema(
    cls: type[Any], _handler: GetCoreSchemaHandler
) -> CoreSchema:
    """Create a Pydantic core schema for a dataclass.

    Generates validation and serialization schema with special handling for
    Literals, Optionals, and Pydantic settings objects.

    Args:
        cls: Dataclass type to create schema for.
        _handler: Pydantic's schema handler (required by protocol, currently unused).

    Returns:
        CoreSchema defining validation and serialization for the dataclass.

    Note:
        Special handling for: Literal types, Optional types (T | None),
        'settings' fields (uses serialize_pydantic_settings), and str types.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    dataclass_fields = []
    field_names = []

    for field in fields(cls):
        field_names.append(field.name)
        field_type = field.type

        field_schema: CoreSchema
        if get_origin(field_type) is Literal:
            field_schema = core_schema.literal_schema(list(get_args(field_type)))
        elif get_origin(field_type) is UnionType:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                if non_none_type is str:
                    field_schema = core_schema.union_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.none_schema(),
                        ]
                    )
                else:
                    field_schema = core_schema.any_schema()
            else:
                field_schema = core_schema.any_schema()
        elif field_type is str:
            field_schema = core_schema.str_schema()
        elif field.name == "settings":
            field_schema = core_schema.any_schema(
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize_pydantic_settings, when_used="json"
                )
            )
        else:
            field_schema = core_schema.any_schema()

        dataclass_fields.append(
            core_schema.dataclass_field(
                name=field.name,
                schema=field_schema,
                kw_only=False,
            )
        )

    return core_schema.dataclass_schema(
        cls,
        core_schema.dataclass_args_schema(cls.__name__, dataclass_fields),
        fields=field_names,
    )

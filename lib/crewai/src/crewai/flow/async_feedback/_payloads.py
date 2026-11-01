"""Data-only codecs for persisted dispatch payloads.

Types come exclusively from the flow's own annotations and built-in feedback
types. Persisted data cannot select a module to import or an object to execute.
"""

from collections.abc import Callable, Iterable
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
import inspect
from pathlib import Path
from typing import Any, get_args, get_type_hints
from uuid import UUID

from pydantic import BaseModel, TypeAdapter


def payload_types(
    methods: Iterable[Callable[..., Any]], extra_types: Iterable[type] = ()
) -> dict[str, type]:
    """Collect trusted model/dataclass/enum types from callable annotations."""
    result: dict[str, type] = {
        f"{cls.__module__}:{cls.__qualname__}": cls
        for cls in (date, datetime, time, timedelta, Decimal, UUID)
    }
    seen: set[int] = set()

    def collect(annotation: Any) -> None:
        if id(annotation) in seen:
            return
        seen.add(id(annotation))
        for arg in get_args(annotation):
            collect(arg)
        if not isinstance(annotation, type):
            return
        if issubclass(annotation, (BaseModel, Enum)) or is_dataclass(annotation):
            result[f"{annotation.__module__}:{annotation.__qualname__}"] = annotation
            if issubclass(annotation, BaseModel):
                for field in annotation.model_fields.values():
                    collect(field.annotation)
            elif is_dataclass(annotation):
                for field_type in get_type_hints(annotation).values():
                    collect(field_type)

    for cls in extra_types:
        collect(cls)
    for method in methods:
        original = inspect.unwrap(method)
        try:
            annotations = get_type_hints(original)
        except (NameError, TypeError):
            # Concrete annotations remain usable even when an unrelated
            # forward reference cannot be resolved in a local function.
            annotations = getattr(original, "__annotations__", {})
        for annotation in annotations.values():
            collect(annotation)
    return result


def encode_payload(value: Any, types: dict[str, type]) -> dict[str, Any]:
    """Encode supported data while retaining declared runtime types.

    Report unsupported values without replacing an already-issued request.
    Every container is tagged so ordinary user keys cannot collide with tags.
    """
    active: set[int] = set()

    def encode(item: Any) -> dict[str, Any]:
        if type(item) in (type(None), bool, int, float, str):
            return {"kind": "scalar", "data": item}
        if isinstance(item, Path):
            return {"kind": "path", "data": str(item)}
        if id(item) in active:
            raise ValueError("Cyclic deferred payloads cannot be persisted")
        active.add(id(item))
        try:
            if type(item) is dict:
                return {
                    "kind": "dict",
                    "data": [[encode(k), encode(v)] for k, v in item.items()],
                }
            if type(item) in (list, tuple, set, frozenset):
                return {"kind": type(item).__name__, "data": [encode(v) for v in item]}
            key = f"{type(item).__module__}:{type(item).__qualname__}"
            if types.get(key) is not type(item):
                raise TypeError(
                    f"Deferred payload type {key} must be declared in a flow method annotation"
                )
            if isinstance(item, BaseModel):
                if item.__pydantic_private__ or any(
                    field.exclude for field in type(item).model_fields.values()
                ):
                    raise TypeError(
                        "Private or excluded model data needs an explicit persistence adapter"
                    )
                data = {name: getattr(item, name) for name in type(item).model_fields}
                if item.__pydantic_extra__:
                    data.update(item.__pydantic_extra__)
            elif is_dataclass(item):
                if any(not field.init for field in fields(item)):
                    raise TypeError(
                        "Dataclass non-init fields need an explicit persistence adapter"
                    )
                data = {
                    field.name: getattr(item, field.name)
                    for field in fields(item)
                    if field.init
                }
            elif isinstance(item, Enum):
                data = item.value
            else:
                data = TypeAdapter(type(item)).dump_python(item, mode="json")
            return {"kind": "typed", "type": key, "data": encode(data)}
        finally:
            active.remove(id(item))

    return encode(value)


def decode_payload(value: dict[str, Any], types: dict[str, type]) -> Any:
    """Restore data using the current flow's trusted type registry only."""
    kind = value["kind"]
    data = value["data"]
    if kind == "scalar":
        if type(data) not in (type(None), bool, int, float, str):
            raise ValueError("Invalid deferred scalar payload")
        return data
    if kind == "path":
        if not isinstance(data, str):
            raise ValueError("Invalid deferred path payload")
        return Path(data)
    if kind == "dict":
        return {decode_payload(k, types): decode_payload(v, types) for k, v in data}
    containers = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
    if kind in containers:
        return containers[kind](decode_payload(v, types) for v in data)
    if kind == "typed":
        cls = types.get(value["type"])
        if cls is None:
            raise ValueError("Persisted payload type is not declared by this flow")
        return TypeAdapter(cls).validate_python(
            decode_payload(data, types), by_name=True
        )
    raise ValueError(f"Unknown deferred payload kind: {kind}")

"""Serializable callable type for Pydantic models.

All callables (named functions, lambdas, closures, methods) are serialized
via ``cloudpickle`` + base64.  On deserialization the base64 payload is
decoded and unpickled back into a live callable.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Annotated, Any

import cloudpickle  # type: ignore[import-untyped]
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


def _deserialize_callable(v: str | Callable[..., Any]) -> Callable[..., Any]:
    """Deserialize a base64-encoded cloudpickle payload, or pass through if already callable."""
    if isinstance(v, str):
        obj = cloudpickle.loads(base64.b85decode(v))
        if not callable(obj):
            raise ValueError(
                f"Deserialized object is {type(obj).__name__}, not a callable"
            )
        return obj  # type: ignore[no-any-return]
    return v


def _serialize_callable(v: Callable[..., Any]) -> str:
    """Serialize any callable to a base64-encoded cloudpickle payload."""
    return base64.b85encode(cloudpickle.dumps(v)).decode("ascii")


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(_deserialize_callable),
    PlainSerializer(_serialize_callable, return_type=str),
    WithJsonSchema({"type": "string"}),
]

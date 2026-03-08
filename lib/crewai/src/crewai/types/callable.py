"""Serializable callable type for Pydantic models.

All callables (ex., named functions, lambdas, closures, methods) are serialized
via ``cloudpickle`` + base64.  On deserialization the base64 payload is
decoded and unpickled back into a live callable.

Deserialization is **opt-in** to prevent arbitrary code execution from
untrusted payloads.  Callers must use :data:`allow_pickle_deserialization` to enable it::

    with allow_pickle_deserialization:q
        task = Task.model_validate_json(untrusted_json)

``cloudpickle`` is an optional dependency.  Serialization and deserialization
will raise ``RuntimeError`` if it is not installed.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from contextvars import ContextVar
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


_ALLOW_PICKLE: ContextVar[bool] = ContextVar("_ALLOW_PICKLE", default=False)


def _import_cloudpickle() -> Any:
    try:
        import cloudpickle  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        raise RuntimeError(
            "cloudpickle is required for callable serialization. "
            "Install it with: uv add 'crewai[pickling]'"
        ) from None
    return cloudpickle


class _AllowPickleDeserialization:
    """Reentrant context manager that opts in to cloudpickle deserialization.

    Usage::

        with allow_pickle_deserialization:
            task = Task.model_validate_json(payload)
    """

    def __enter__(self) -> None:
        self._token = _ALLOW_PICKLE.set(True)

    def __exit__(self, *_: object) -> None:
        _ALLOW_PICKLE.reset(self._token)


allow_pickle_deserialization = _AllowPickleDeserialization()


def _deserialize_callable(v: str | Callable[..., Any]) -> Callable[..., Any]:
    """Deserialize a base64-encoded cloudpickle payload, or pass through if already callable."""
    if isinstance(v, str):
        if not _ALLOW_PICKLE.get():
            raise RuntimeError(
                "Refusing to unpickle a callable from untrusted data. "
                "Wrap the deserialization call with "
                "`with allow_pickle_deserialization: ...` "
                "if you trust the source."
            )
        cloudpickle = _import_cloudpickle()
        obj = cloudpickle.loads(base64.b85decode(v))
        if not callable(obj):
            raise ValueError(
                f"Deserialized object is {type(obj).__name__}, not a callable"
            )
        return obj  # type: ignore[no-any-return]
    return v


def _serialize_callable(v: Callable[..., Any]) -> str:
    """Serialize any callable to a base64-encoded cloudpickle payload."""
    cloudpickle = _import_cloudpickle()
    return base64.b85encode(cloudpickle.dumps(v)).decode("ascii")


SerializableCallable = Annotated[
    Callable[..., Any],
    BeforeValidator(_deserialize_callable),
    PlainSerializer(_serialize_callable, return_type=str),
    WithJsonSchema({"type": "string"}),
]

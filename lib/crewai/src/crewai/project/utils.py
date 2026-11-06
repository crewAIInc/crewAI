"""Utility functions for the crewai project module."""

from collections.abc import Callable, Coroutine
from functools import wraps
import inspect
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel


P = ParamSpec("P")
R = TypeVar("R")

_MEMO_STORE_ATTR = "__crewai_memo__"


def _make_hashable(arg: Any) -> Any:
    """Convert argument to hashable form for caching.

    Args:
        arg: The argument to convert.

    Returns:
        Hashable representation of the argument.
    """
    if isinstance(arg, BaseModel):
        return arg.model_dump_json()
    if isinstance(arg, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in arg.items()))
    if isinstance(arg, list):
        return tuple(_make_hashable(item) for item in arg)
    if hasattr(arg, "__dict__"):
        return ("__instance__", id(arg))
    return arg


def _instance_memo_store(target: Any) -> dict[tuple[str, str], Any] | None:
    """Return the per-instance memo store, or None when not applicable.

    Args:
        target: First positional argument of the wrapped call (the bound
            instance for decorated methods).

    Returns:
        The instance's memo dict, created on first use, or None for calls
        without an instance (or instances that reject attribute writes), in
        which case the call runs uncached.
    """
    if target is None or not hasattr(target, "__dict__"):
        return None
    store = target.__dict__.get(_MEMO_STORE_ATTR)
    if not isinstance(store, dict):
        store = {}
        try:
            target.__dict__[_MEMO_STORE_ATTR] = store
        except (AttributeError, TypeError):
            return None
    return cast("dict[tuple[str, str], Any]", store)


def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a method, storing results on the bound instance.

    Handles both sync and async methods. Pydantic BaseModel instances are
    converted to JSON strings before hashing for cache lookup.

    The store lives on the instance so it dies with it: a module-level store
    keyed by ``id(self)`` handed a recycled id's cached result to a fresh
    instance after garbage collection.

    Args:
        meth: The method to memoize.

    Returns:
        A memoized version of the method that caches results.
    """
    if inspect.iscoroutinefunction(meth):
        return cast(Callable[P, R], _memoize_async(meth))
    return _memoize_sync(meth)


def _memoize_sync(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a synchronous method."""

    @wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        store = _instance_memo_store(args[0]) if args else None
        if store is None:
            return meth(*args, **kwargs)

        hashable_args = tuple(_make_hashable(arg) for arg in args[1:])
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items())
        )
        cache_key = (meth.__name__, str((hashable_args, hashable_kwargs)))

        if cache_key in store:
            return cast("R", store[cache_key])

        result = meth(*args, **kwargs)
        store[cache_key] = result
        return result

    return cast(Callable[P, R], wrapper)


def _memoize_async(
    meth: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Memoize an async method."""

    @wraps(meth)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        store = _instance_memo_store(args[0]) if args else None
        if store is None:
            return await meth(*args, **kwargs)

        hashable_args = tuple(_make_hashable(arg) for arg in args[1:])
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items())
        )
        cache_key = (meth.__name__, str((hashable_args, hashable_kwargs)))

        if cache_key in store:
            return cast("R", store[cache_key])

        result = await meth(*args, **kwargs)
        store[cache_key] = result
        return result

    return wrapper

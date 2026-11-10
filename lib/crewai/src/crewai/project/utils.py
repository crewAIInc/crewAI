"""Utility functions for the crewai project module."""

from collections.abc import Callable, Coroutine
from functools import wraps
import inspect
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel

from crewai.agents.cache.cache_handler import CacheHandler


P = ParamSpec("P")
R = TypeVar("R")
_CACHE_ATTR = "_memoize_cache"


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


def _instance_cache(args: tuple[Any, ...]) -> CacheHandler:
    """Return the cache of the instance a memoized method is called on.

    Results live on the instance rather than in a module-level cache keyed by
    ``id(self)``: ids are reused once an instance is freed, which handed a new
    crew the agents and tasks of an unrelated, already-collected one.
    """
    namespace = vars(args[0])
    cache = namespace.get(_CACHE_ATTR)
    if not isinstance(cache, CacheHandler):
        cache = namespace[_CACHE_ATTR] = CacheHandler()
    return cache


def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a method by caching its results per instance, based on arguments.

    Handles both sync and async methods. Pydantic BaseModel instances are
    converted to JSON strings before hashing for cache lookup.

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
        cache = _instance_cache(args)
        hashable_args = tuple(_make_hashable(arg) for arg in args[1:])
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items())
        )
        cache_key = str((hashable_args, hashable_kwargs))

        cached_result: R | None = cache.read(tool=meth.__name__, tool_input=cache_key)
        if cached_result is not None:
            return cached_result

        result = meth(*args, **kwargs)
        cache.add(tool=meth.__name__, tool_input=cache_key, output=result)
        return result

    return cast(Callable[P, R], wrapper)


def _memoize_async(
    meth: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Memoize an async method."""

    @wraps(meth)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        cache = _instance_cache(args)
        hashable_args = tuple(_make_hashable(arg) for arg in args[1:])
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items())
        )
        cache_key = str((hashable_args, hashable_kwargs))

        cached_result: R | None = cache.read(tool=meth.__name__, tool_input=cache_key)
        if cached_result is not None:
            return cached_result

        result = await meth(*args, **kwargs)
        cache.add(tool=meth.__name__, tool_input=cache_key, output=result)
        return result

    return wrapper

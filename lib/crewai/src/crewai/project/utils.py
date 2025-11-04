"""Utility functions for the crewai project module."""

from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel


P = ParamSpec("P")
R = TypeVar("R")


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
    return arg


def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a method by caching its results based on arguments.

    Handles Pydantic BaseModel instances by converting them to JSON strings
    before hashing for cache lookup.

    Args:
        meth: The method to memoize.

    Returns:
        A memoized version of the method that caches results.
    """

    @wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Wrapper that converts arguments to hashable form before caching.

        Args:
            *args: Positional arguments to the memoized method.
            **kwargs: Keyword arguments to the memoized method.

        Returns:
            The result of the memoized method call.
        """
        hashable_args = tuple(_make_hashable(arg) for arg in args)
        hashable_kwargs = tuple(
            sorted((k, _make_hashable(v)) for k, v in kwargs.items())
        )

        @lru_cache(typed=True)
        def _cached(
            h_args: tuple[Any, ...], h_kwargs: tuple[tuple[str, Any], ...]
        ) -> R:
            """Internal cache function that stores results for hashable arguments.

            Args:
                h_args: Positional arguments in hashable form.
                h_kwargs: Keyword arguments in hashable form as sorted tuple.

            Returns:
                The cached result of the memoized method.
            """
            return meth(*args, **kwargs)

        return _cached(hashable_args, hashable_kwargs)

    return cast(Callable[P, R], wrapper)

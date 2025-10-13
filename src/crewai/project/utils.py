"""Utility functions for the crewai project module."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a method by caching its results based on arguments.

    Args:
        meth: The method to memoize.

    Returns:
        A memoized version of the method that caches results.
    """
    cache: dict[Any, R] = {}

    @wraps(meth)
    def memoized_func(*args: P.args, **kwargs: P.kwargs) -> R:
        """Memoized wrapper method.

        Args:
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The cached or computed result of the method.
        """
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = meth(*args, **kwargs)
        return cache[key]

    return memoized_func

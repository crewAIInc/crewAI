"""Utility functions for the crewai project module."""

from collections.abc import Callable
from functools import lru_cache
from typing import ParamSpec, TypeVar, cast


P = ParamSpec("P")
R = TypeVar("R")


def memoize(meth: Callable[P, R]) -> Callable[P, R]:
    """Memoize a method by caching its results based on arguments.

    Args:
        meth: The method to memoize.

    Returns:
        A memoized version of the method that caches results.
    """
    return cast(Callable[P, R], lru_cache(typed=True)(meth))

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def memoize(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator that caches function results based on arguments.

    Args:
        func: The function to memoize.

    Returns:
        The memoized function.
    """
    cache: dict[tuple, R] = {}

    @wraps(func)
    def memoized_func(*args: P.args, **kwargs: P.kwargs) -> R:
        """Memoized wrapper function."""
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func

from typing import Optional, TypeVar

T = TypeVar("T")


def assert_not_none(value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Expected non-None value")
    return value

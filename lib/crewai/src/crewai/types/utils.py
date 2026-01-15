"""Utilities for creating and manipulating types."""

from typing import Annotated, Final, Literal, cast


_DYNAMIC_LITERAL_ALIAS: Final[Literal["DynamicLiteral"]] = "DynamicLiteral"


def create_literals_from_strings(
    values: Annotated[
        tuple[str, ...], "Should contain unique strings; duplicates will be removed"
    ],
) -> type:
    """Create a Literal type for each A2A agent ID.

    Args:
        values: a tuple of the A2A agent IDs

    Returns:
        Literal type for each A2A agent ID

    Raises:
        ValueError: If values is empty (Literal requires at least one value)
    """
    unique_values: tuple[str, ...] = tuple(dict.fromkeys(values))
    if not unique_values:
        raise ValueError("Cannot create Literal type from empty values")
    return cast(type, Literal.__getitem__(unique_values))

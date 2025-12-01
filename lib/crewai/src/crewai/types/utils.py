"""Utilities for creating and manipulating types."""

from typing import Annotated, Final, Literal

from typing_extensions import TypeAliasType


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
    """
    unique_values: tuple[str, ...] = tuple(dict.fromkeys(values))
    return Literal.__getitem__(unique_values)

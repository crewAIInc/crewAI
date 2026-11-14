"""Confidence-aware route selector for Jev Choice results."""

from __future__ import annotations

from collections.abc import Collection
import math


def select_jev_route(
    choice: str | None,
    confidence: float | None,
    allowed_routes: Collection[str],
    min_confidence: float,
    fallback: str,
) -> str:
    """Map a Jev Choice answer onto caller-provided Flow route labels using confidence thresholding.

    TypeSafe confidence semantics provide Choice answers with a confidence value
    ranging from 0.0 to 1.0. This helper uses the supplied value directly to determine
    whether to route to the selected choice or fall back to the configured fallback route.

    Routing rules:
        1. A known choice (present in `allowed_routes`) whose confidence meets or exceeds
           `min_confidence` selects that route.
        2. Low confidence (< min_confidence), unknown choice label, missing choice (None/empty),
           or missing/non-finite/out-of-range confidence selects the explicit `fallback`.
        3. Invalid configuration (invalid threshold, fallback not in allowed routes,
           or empty allowed routes) raises a clear ValueError or TypeError.

    Args:
        choice: Selected choice label from Jev Choice result, or None.
        confidence: Confidence score associated with the choice (expected 0.0 to 1.0), or None.
        allowed_routes: Collection of valid route labels recognized by the Flow router.
        min_confidence: Minimum confidence threshold required to accept the selected choice.
            Must be a finite number between 0.0 and 1.0 inclusive.
        fallback: Explicit fallback route label to use if confidence is low, choice is
            missing/unknown, or confidence value is invalid. Must be in `allowed_routes`.

    Returns:
        str: Either `choice` if valid and confidence meets the threshold, or `fallback`.

    Raises:
        ValueError: If configuration is invalid:
            - `min_confidence` is outside [0.0, 1.0] or non-finite.
            - `allowed_routes` is empty or contains non-string/empty items.
            - `fallback` is not in `allowed_routes`.
        TypeError: If configuration parameters are of invalid types.
    """
    # --- Configuration validation ---
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        raise TypeError(
            f"min_confidence must be a float or int, got {type(min_confidence).__name__}."
        )

    if not math.isfinite(min_confidence):
        raise ValueError(
            f"min_confidence must be a finite number, got {min_confidence}."
        )

    min_conf = float(min_confidence)
    if not (0.0 <= min_conf <= 1.0):
        raise ValueError(
            f"min_confidence must be between 0.0 and 1.0 inclusive, got {min_conf}."
        )

    if not isinstance(allowed_routes, Collection) or isinstance(allowed_routes, (str, bytes)):
        raise TypeError(
            f"allowed_routes must be a collection of route labels, got {type(allowed_routes).__name__}."
        )

    if len(allowed_routes) == 0:
        raise ValueError("allowed_routes cannot be empty.")

    for r in allowed_routes:
        if not isinstance(r, str) or not r.strip():
            raise ValueError(
                f"allowed_routes elements must be non-empty strings, got {r!r}."
            )

    routes_set = set(allowed_routes)

    if not isinstance(fallback, str):
        raise TypeError(
            f"fallback must be a string, got {type(fallback).__name__}."
        )

    if fallback not in routes_set:
        raise ValueError(
            f"fallback route '{fallback}' must be present in allowed_routes: {sorted(routes_set)}."
        )

    # --- Runtime decision evaluation ---
    # Validate choice
    if choice is None or not isinstance(choice, str) or choice not in routes_set:
        return fallback

    # Validate confidence
    if confidence is None or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return fallback

    if not math.isfinite(confidence):
        return fallback

    conf = float(confidence)
    if conf < 0.0 or conf > 1.0:
        return fallback

    # Check threshold: confidence >= min_confidence selects choice, else fallback
    if conf >= min_conf:
        return choice

    return fallback


# Provide alias for convenience and discoverability
select_choice_route = select_jev_route

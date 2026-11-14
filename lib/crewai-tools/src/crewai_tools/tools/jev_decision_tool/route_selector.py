"""Confidence-aware route selection for Jev Choice results.

Maps a Jev Choice answer onto a caller-provided set of Flow route labels.
The confidence is TypeSafe's reported confidence for the selected choice, a
value in ``[0, 1]`` (https://docs.typesafe.ai/confidence). It is used exactly
as supplied and is never recomputed.

Example, inside a Flow (``classify`` returns a synthetic decision dict shaped
like ``JevDecisionTool``'s result: ``{"choice": ..., "confidence": ...}``)::

    from crewai_tools.tools.jev_decision_tool.route_selector import select_route


    @router(classify)
    def route(self, decision: dict[str, Any]) -> str:
        return select_route(
            decision.get("choice"),
            decision.get("confidence"),
            allowed_routes=("refund", "tech_support", "triage"),
            fallback="triage",
            min_confidence=0.8,
        )
"""

from __future__ import annotations

from collections.abc import Collection


def select_route(
    choice: str | None,
    confidence: float | None,
    *,
    allowed_routes: Collection[str],
    fallback: str,
    min_confidence: float,
) -> str:
    """Return the Flow route label for a Jev Choice answer.

    Args:
        choice: The selected Choice label, or ``None`` if absent.
        confidence: TypeSafe's confidence for ``choice``, or ``None`` if absent.
        allowed_routes: The route labels the Flow router may return.
        fallback: The route returned when the answer cannot be trusted.
        min_confidence: The smallest confidence that still selects ``choice``.

    Returns:
        ``choice`` when it is one of ``allowed_routes`` and ``confidence`` is an
        ``int`` or ``float`` (not ``bool``) in ``[0, 1]`` that is at least
        ``min_confidence``; otherwise ``fallback``. So ``None``, NaN, infinities,
        out-of-range values, non-numbers and labels outside ``allowed_routes``
        all fall back rather than raise.

    Raises:
        ValueError: If ``allowed_routes`` is a single string, ``fallback`` is
            not one of ``allowed_routes``, or ``min_confidence`` is not an
            ``int`` or ``float`` (not ``bool``) in ``[0, 1]``.
    """
    if isinstance(allowed_routes, str):
        # A str is a Collection[str] of its characters; membership would be
        # substring matching and could return a label that is not a route.
        raise ValueError("allowed_routes must be a collection of labels, not a string")
    if fallback not in allowed_routes:
        raise ValueError(
            f"fallback {fallback!r} is not one of allowed_routes {allowed_routes!r}"
        )
    if not _is_unit_interval(min_confidence):
        raise ValueError(
            f"min_confidence must be an int or float in [0, 1], got {min_confidence!r}"
        )

    if not isinstance(choice, str) or choice not in allowed_routes:
        return fallback
    # ``is None`` is redundant at runtime but narrows ``float | None`` for mypy.
    if confidence is None or not _is_unit_interval(confidence):
        return fallback
    return choice if confidence >= min_confidence else fallback


def _is_unit_interval(value: object) -> bool:
    """Return whether ``value`` is an ``int`` or ``float`` (not ``bool``) in ``[0, 1]``.

    NaN and the infinities fail the range comparison, so no separate
    finiteness check is needed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= value <= 1.0

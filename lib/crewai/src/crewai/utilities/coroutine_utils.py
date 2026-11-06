"""Coroutine detection helpers.

``asyncio.iscoroutinefunction`` is deprecated since Python 3.14 (scheduled for
removal in 3.16); ``inspect.iscoroutinefunction`` is the documented replacement.

The ``inspect`` variant, however, does not recognise wrappers that only carry the
legacy ``_is_coroutine`` marker (as set by
``crewai.flow.flow_wrappers.FlowMethod`` on Python < 3.12, where
``inspect.markcoroutinefunction`` is unavailable). ``asyncio``'s deprecated
function does recognise them, so both checks are combined here.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


_LEGACY_COROUTINE_MARKER = getattr(asyncio.coroutines, "_is_coroutine", None)


def is_coroutine_function(func: Any) -> bool:
    """Return True if *func* is a coroutine function, including legacy wrappers.

    Checks ``inspect.iscoroutinefunction`` first (the documented replacement for
    the deprecated ``asyncio.iscoroutinefunction``) and then the legacy
    ``_is_coroutine`` marker that ``FlowMethod`` sets on Python < 3.12.
    """
    if inspect.iscoroutinefunction(func):
        return True
    marker = getattr(func, "_is_coroutine", None)
    return marker is not None and marker is _LEGACY_COROUTINE_MARKER

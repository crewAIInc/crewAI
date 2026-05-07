"""Re-export of ``crewai_core.plus_api.PlusAPI``.

Kept as a stable import path for the framework; new code should import from
``crewai_core.plus_api`` directly.
"""

from __future__ import annotations

from crewai_core.plus_api import PlusAPI as PlusAPI


__all__ = ["PlusAPI"]

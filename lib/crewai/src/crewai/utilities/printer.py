"""Deprecated: use ``crewai_core.printer`` instead."""

from __future__ import annotations

import warnings

from crewai_core.printer import (
    PRINTER as PRINTER,
    ColoredText as ColoredText,
    Printer as Printer,
    PrinterColor as PrinterColor,
)


__all__ = ["PRINTER", "ColoredText", "Printer", "PrinterColor"]


warnings.warn(
    "crewai.utilities.printer is deprecated; import from crewai_core.printer.",
    DeprecationWarning,
    stacklevel=2,
)

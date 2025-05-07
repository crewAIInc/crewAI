"""
Backward compatibility module for crewai.telemtry to handle typo in import statements.

This module allows older code that imports from `crewai.telemtry` (misspelled)
to continue working by re-exporting the Telemetry class from the correctly
spelled `crewai.telemetry` module.
"""
import warnings

from crewai.telemetry import Telemetry

warnings.warn(
    "Importing from 'crewai.telemtry' is deprecated due to spelling issues. "
    "Please use 'from crewai.telemetry import Telemetry' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["Telemetry"]

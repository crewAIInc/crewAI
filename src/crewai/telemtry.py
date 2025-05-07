"""
Backward compatibility module for crewai.telemtry to handle typo in import statements.

This module allows older code that imports from `crewai.telemtry` (misspelled)
to continue working by re-exporting the Telemetry class from the correctly
spelled `crewai.telemetry` module.
"""
from crewai.telemetry import Telemetry

__all__ = ["Telemetry"]

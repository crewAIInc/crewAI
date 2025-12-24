"""Kraken Futures Assignment Program Tools - Private endpoints voor assignment programma."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Lijst Assignment Programma's
# =============================================================================
class KrakenFuturesListAssignmentProgramsTool(KrakenFuturesBaseTool):
    """Lijst beschikbare assignment programma's."""

    name: str = "kraken_futures_list_assignment_programs"
    description: str = "Haal lijst op van beschikbare assignment programma's en hun details."

    def _run(self) -> str:
        """Haal assignment programma's op van Kraken Futures."""
        result = self._private_request("assignmentprograms", method="GET")
        return str(result)


# =============================================================================
# Tool 2: Voeg Assignment Voorkeur Toe
# =============================================================================
class KrakenFuturesAddAssignmentPreferenceInput(BaseModel):
    """Input schema voor KrakenFuturesAddAssignmentPreferenceTool."""

    program_id: str = Field(..., description="Assignment programma ID")
    symbol: str = Field(..., description="Futures symbool")
    preference: str = Field(
        ..., description="Voorkeur instelling: 'assign', 'noassign'"
    )


class KrakenFuturesAddAssignmentPreferenceTool(KrakenFuturesBaseTool):
    """Voeg assignment voorkeur toe."""

    name: str = "kraken_futures_add_assignment_preference"
    description: str = "Voeg een assignment voorkeur toe voor een specifiek programma en symbool."
    args_schema: type[BaseModel] = KrakenFuturesAddAssignmentPreferenceInput

    def _run(self, program_id: str, symbol: str, preference: str) -> str:
        """Voeg assignment voorkeur toe op Kraken Futures."""
        data = {
            "programId": program_id,
            "symbol": symbol,
            "preference": preference,
        }
        result = self._private_request("assignmentprograms/preferences", data)
        return str(result)


# =============================================================================
# Tool 3: Verwijder Assignment Voorkeur
# =============================================================================
class KrakenFuturesDeleteAssignmentPreferenceInput(BaseModel):
    """Input schema voor KrakenFuturesDeleteAssignmentPreferenceTool."""

    program_id: str = Field(..., description="Assignment programma ID")
    symbol: str = Field(..., description="Futures symbool")


class KrakenFuturesDeleteAssignmentPreferenceTool(KrakenFuturesBaseTool):
    """Verwijder assignment voorkeur."""

    name: str = "kraken_futures_delete_assignment_preference"
    description: str = "Verwijder een assignment voorkeur voor een specifiek programma en symbool."
    args_schema: type[BaseModel] = KrakenFuturesDeleteAssignmentPreferenceInput

    def _run(self, program_id: str, symbol: str) -> str:
        """Verwijder assignment voorkeur op Kraken Futures."""
        data = {
            "programId": program_id,
            "symbol": symbol,
        }
        # Use DELETE method conceptually, implemented via POST with action
        result = self._private_request(
            "assignmentprograms/preferences/delete", data
        )
        return str(result)


# =============================================================================
# Tool 4: Lijst Assignment Geschiedenis
# =============================================================================
class KrakenFuturesListAssignmentHistoryInput(BaseModel):
    """Input schema voor KrakenFuturesListAssignmentHistoryTool."""

    program_id: str | None = Field(
        default=None, description="Filter op programma ID"
    )
    symbol: str | None = Field(
        default=None, description="Filter op symbool"
    )


class KrakenFuturesListAssignmentHistoryTool(KrakenFuturesBaseTool):
    """Lijst assignment geschiedenis."""

    name: str = "kraken_futures_list_assignment_history"
    description: str = "Haal geschiedenis op van assignment events inclusief bedragen en timestamps."
    args_schema: type[BaseModel] = KrakenFuturesListAssignmentHistoryInput

    def _run(
        self, program_id: str | None = None, symbol: str | None = None
    ) -> str:
        """Haal assignment geschiedenis op van Kraken Futures."""
        data = {}
        if program_id:
            data["programId"] = program_id
        if symbol:
            data["symbol"] = symbol
        result = self._private_request("assignmentprograms/history", data, method="GET")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesListAssignmentProgramsTool",
    "KrakenFuturesAddAssignmentPreferenceTool",
    "KrakenFuturesDeleteAssignmentPreferenceTool",
    "KrakenFuturesListAssignmentHistoryTool",
]

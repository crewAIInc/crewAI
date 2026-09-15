"""Pydantic input schemas for the Live Tennis API tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


LiveTennisAction = Literal[
    "live_matches",
    "upcoming_matches",
    "fixtures",
    "search_players",
    "player_profile",
    "rankings",
    "usage",
]
"""Operations the tool can perform, each mapping to one REST endpoint."""

RankingSystem = Literal["atp", "wta", "itf_jt", "itf_mt", "itf_wt"]
"""Ranking systems with a published rank-ordered listing.

UTR is intentionally absent: the API's ``/rankings`` listing mode does not
serve it ("`utr` has no listing — it is a rating, not a ranking").
"""

Tour = Literal["atp", "wta", "challenger", "itf", "juniors"]
"""Valid ``tour`` filter values; an unrecognised value is a 400 from the API."""


class LiveTennisToolSchema(BaseModel):
    """Input schema for LiveTennisTool."""

    action: LiveTennisAction = Field(
        ...,
        description=(
            "Operation to perform: 'live_matches' (matches in play right now, "
            "with current scores), 'upcoming_matches' (matches starting soon), "
            "'fixtures' (scheduled matches), 'search_players' (find players by "
            "name, requires 'search'), 'player_profile' (single player detail "
            "including current ranking, requires 'player_id'), 'rankings' "
            "(published ranking table, requires 'system'; needs a PRO-plan "
            "API key), 'usage' (your API quota and consumption)."
        ),
    )
    tour: Tour | None = Field(
        default=None,
        description=(
            "Optional tour filter for match and fixture actions: 'atp', 'wta', "
            "'challenger', 'itf' or 'juniors'. Omit for all tours."
        ),
    )
    search: str | None = Field(
        default=None,
        description=(
            "Player name (or name fragment) to look up. Required for the "
            "'search_players' action."
        ),
    )
    player_id: int | None = Field(
        default=None,
        description=(
            "Numeric player id as returned by 'search_players'. Required for "
            "the 'player_profile' action."
        ),
    )
    system: RankingSystem | None = Field(
        default=None,
        description=(
            "Ranking system for the 'rankings' action: 'atp', 'wta', 'itf_jt', "
            "'itf_mt' or 'itf_wt'. The rankings listing needs a PRO-plan key."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of results to return (API default: 50).",
    )
    offset: int | None = Field(
        default=None,
        ge=0,
        description="Pagination offset into the result list.",
    )

    @model_validator(mode="after")
    def _validate_action_arguments(self) -> LiveTennisToolSchema:
        """Ensure the arguments each action depends on are present."""
        if self.action == "search_players" and not (
            self.search and self.search.strip()
        ):
            raise ValueError("'search' is required for the 'search_players' action.")
        if self.action == "player_profile" and self.player_id is None:
            raise ValueError("'player_id' is required for the 'player_profile' action.")
        if self.action == "rankings" and self.system is None:
            raise ValueError("'system' is required for the 'rankings' action.")
        return self

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

RankingSystem = Literal["atp", "wta", "itf_jt", "itf_mt", "itf_wt", "utr"]


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
            "(published ranking table, requires 'system'), 'usage' (your API "
            "quota and consumption)."
        ),
    )
    tour: str | None = Field(
        default=None,
        description=(
            "Optional tour filter for match and fixture actions, e.g. 'atp' or 'wta'."
        ),
    )
    search: str | None = Field(
        default=None,
        description=(
            "Player name (or name fragment) to look up. Required for the "
            "'search_players' action."
        ),
    )
    player_id: str | None = Field(
        default=None,
        description=(
            "Player id as returned by 'search_players'. Required for the "
            "'player_profile' action."
        ),
    )
    system: RankingSystem | None = Field(
        default=None,
        description=(
            "Ranking system for the 'rankings' action: 'atp', 'wta', 'itf_jt', "
            "'itf_mt', 'itf_wt' or 'utr'."
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
        if self.action == "search_players" and not (
            self.search and self.search.strip()
        ):
            raise ValueError("'search' is required for the 'search_players' action.")
        if self.action == "player_profile" and not (
            self.player_id and self.player_id.strip()
        ):
            raise ValueError("'player_id' is required for the 'player_profile' action.")
        if self.action == "rankings" and self.system is None:
            raise ValueError("'system' is required for the 'rankings' action.")
        return self

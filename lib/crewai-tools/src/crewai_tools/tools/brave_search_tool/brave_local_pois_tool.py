from typing import Any, cast

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.response_types import LocalPOIs
from crewai_tools.tools.brave_search_tool.schemas import (
    LocalPOIsDescriptionHeaders,
    LocalPOIsDescriptionParams,
    LocalPOIsHeaders,
    LocalPOIsParams,
)


DayOpeningHours = LocalPOIs.DayOpeningHours
OpeningHours = LocalPOIs.OpeningHours
LocationResult = LocalPOIs.LocationResult
LocalPOIsResponse = LocalPOIs.Response


def _flatten_slots(slots: list[DayOpeningHours]) -> list[dict[str, str]]:
    """Convert a list of DayOpeningHours dicts into simplified entries."""
    return [
        {
            "day": slot["full_name"].lower(),
            "opens": slot["opens"],
            "closes": slot["closes"],
        }
        for slot in slots
    ]


def _simplify_opening_hours(result: LocationResult) -> list[dict[str, str]] | None:
    """Collapse opening_hours into a flat list of {day, opens, closes} dicts."""
    hours = result.get("opening_hours")
    if not hours:
        return None

    entries: list[dict[str, str]] = []

    current = hours.get("current_day")
    if current:
        entries.extend(_flatten_slots(current))

    days = hours.get("days")
    if days:
        for day_slots in days:
            entries.extend(_flatten_slots(day_slots))

    return entries or None


def _format_location_result(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one POI result, tolerating nullable nested objects."""
    postal_address = result.get("postal_address") or {}
    contact = result.get("contact") or {}

    return {
        "title": result.get("title"),
        "url": result.get("url"),
        "description": result.get("description"),
        "address": postal_address.get("displayAddress"),
        "contact": contact.get("telephone") or contact.get("email") or None,
        "opening_hours": _simplify_opening_hours(cast(LocationResult, result)),
    }


class BraveLocalPOIsTool(BraveSearchToolBase):
    """A tool that retrieves local POIs using the Brave Search API."""

    name: str = "Brave Local POIs"
    args_schema: type[BaseModel] = LocalPOIsParams
    header_schema: type[BaseModel] = LocalPOIsHeaders
    description: str = (
        "A tool that retrieves local POIs using the Brave Search API. "
        "Results are returned as structured JSON data."
    )
    search_url: str = "https://api.search.brave.com/res/v1/local/pois"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = response.get("results", [])
        return [_format_location_result(result) for result in results]


class BraveLocalPOIsDescriptionTool(BraveSearchToolBase):
    """A tool that retrieves AI-generated descriptions for local POIs using the Brave Search API."""

    name: str = "Brave Local POI Descriptions"
    args_schema: type[BaseModel] = LocalPOIsDescriptionParams
    header_schema: type[BaseModel] = LocalPOIsDescriptionHeaders
    description: str = (
        "A tool that retrieves AI-generated descriptions for local POIs using the Brave Search API. "
        "Results are returned as structured JSON data."
    )
    search_url: str = "https://api.search.brave.com/res/v1/local/descriptions"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = response.get("results", [])
        return [
            {
                "id": result.get("id"),
                "description": result.get("description"),
            }
            for result in results
        ]

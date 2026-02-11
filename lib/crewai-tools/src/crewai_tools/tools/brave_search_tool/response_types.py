from __future__ import annotations

from typing import Literal, TypedDict


class LocalPOIs:
    class PostalAddress(TypedDict, total=False):
        type: Literal["PostalAddress"]
        country: str
        postalCode: str
        streetAddress: str
        addressRegion: str
        addressLocality: str
        displayAddress: str

    class DayOpeningHours(TypedDict):
        abbr_name: str
        full_name: str
        opens: str
        closes: str

    class OpeningHours(TypedDict, total=False):
        current_day: list[LocalPOIs.DayOpeningHours]
        days: list[list[LocalPOIs.DayOpeningHours]]

    class LocationResult(TypedDict, total=False):
        provider_url: str
        title: str
        url: str
        id: str | None
        opening_hours: LocalPOIs.OpeningHours | None
        postal_address: LocalPOIs.PostalAddress | None

    class Response(TypedDict, total=False):
        type: Literal["local_pois"]
        results: list[LocalPOIs.LocationResult]


class LLMContext:
    class LLMContextItem(TypedDict, total=False):
        snippets: list[str]
        title: str
        url: str

    class LLMContextMapItem(TypedDict, total=False):
        name: str
        snippets: list[str]
        title: str
        url: str

    class LLMContextPOIItem(TypedDict, total=False):
        name: str
        snippets: list[str]
        title: str
        url: str

    class Grounding(TypedDict, total=False):
        generic: list[LLMContext.LLMContextItem]
        poi: LLMContext.LLMContextPOIItem
        map: list[LLMContext.LLMContextMapItem]

    class Sources(TypedDict, total=False):
        pass

    class Response(TypedDict, total=False):
        grounding: LLMContext.Grounding
        sources: LLMContext.Sources

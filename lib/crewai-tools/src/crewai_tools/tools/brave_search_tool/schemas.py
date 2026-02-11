from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.types import StringConstraints


# Common types
Units = Literal["metric", "imperial"]
SafeSearch = Literal["off", "moderate", "strict"]
Freshness = (
    Literal["pd", "pw", "pm", "py"]
    | Annotated[
        str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$")
    ]
)
ResultFilter = list[
    Literal[
        "discussions",
        "faq",
        "infobox",
        "news",
        "query",
        "summarizer",
        "videos",
        "web",
        "locations",
    ]
]


class LLMContextParams(BaseModel):
    """Parameters for Brave LLM Context endpoint."""

    q: str = Field(
        description="Search query to perform",
        min_length=1,
        max_length=400,
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
        pattern=r"^[A-Z]{2}$",
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return. Actual number may be less.",
        ge=1,
        le=50,
    )
    maximum_number_of_urls: int | None = Field(
        default=None,
        description="The maximum number of URLs to include in the context.",
        ge=1,
        le=50,
    )
    maximum_number_of_tokens: int | None = Field(
        default=None,
        description="The approximate maximum number of tokens to include in the context.",
        ge=1,
        le=32768,
    )
    maximum_number_of_snippets: int | None = Field(
        default=None,
        description="The maximum number of different snippets to include in the context.",
        ge=1,
        le=100,
    )
    context_threshold_mode: (
        Literal["disabled", "strict", "lenient", "balanced"] | None
    ) = Field(
        default=None,
        description="The mode to use for the context thresholding.",
    )
    maximum_number_of_tokens_per_url: int | None = Field(
        default=None,
        description="The maximum number of tokens to include for each URL in the context.",
        ge=1,
        le=8192,
    )
    maximum_number_of_snippets_per_url: int | None = Field(
        default=None,
        description="The maximum number of snippets to include per URL.",
        ge=1,
        le=100,
    )
    goggles: str | list[str] | None = Field(
        default=None,
        description="Goggles act as a custom re-ranking mechanism. Goggle source or URLs.",
    )
    enable_local: bool | None = Field(
        default=None,
        description="Whether to enable local recall. Not setting this value means auto-detect and uses local recall if any of the localization headers are provided.",
    )


class WebSearchParams(BaseModel):
    """Parameters for Brave Web Search endpoint."""

    q: str = Field(
        description="Search query to perform",
        min_length=1,
        max_length=400,
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
        pattern=r"^[A-Z]{2}$",
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    ui_lang: str | None = Field(
        default=None,
        description="Language code for the user interface (e.g., 'en-US', 'es-AR').",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return. Actual number may be less.",
        ge=1,
        le=20,
    )
    offset: int | None = Field(
        default=None,
        description="Skip the first N result sets/pages. Max is 9.",
        ge=0,
        le=9,
    )
    safesearch: Literal["off", "moderate", "strict"] | None = Field(
        default=None,
        description="Filter out explicit content. Options: off/moderate/strict",
    )
    spellcheck: bool | None = Field(
        default=None,
        description="Attempt to correct spelling errors in the search query.",
    )
    freshness: Freshness | None = Field(
        default=None,
        description="Enforce freshness of results. Options: pd/pw/pm/py, or YYYY-MM-DDtoYYYY-MM-DD",
    )
    text_decorations: bool | None = Field(
        default=None,
        description="Include markup to highlight search terms in the results.",
    )
    extra_snippets: bool | None = Field(
        default=None,
        description="Include up to 5 text snippets for each page if possible.",
    )
    result_filter: ResultFilter | None = Field(
        default=None,
        description="Filter the results by type. Options: discussions/faq/infobox/news/query/summarizer/videos/web/locations. Note: The `count` parameter is applied only to the `web` results.",
    )
    units: Units | None = Field(
        default=None,
        description="The units to use for the results. Options: metric/imperial",
    )
    goggles: str | list[str] | None = Field(
        default=None,
        description="Goggles act as a custom re-ranking mechanism. Goggle source or URLs.",
    )
    summary: bool | None = Field(
        default=None,
        description="Whether to generate a summarizer ID for the results.",
    )
    enable_rich_callback: bool | None = Field(
        default=None,
        description="Whether to enable rich callbacks for the results. Requires Pro level subscription.",
    )
    include_fetch_metadata: bool | None = Field(
        default=None,
        description="Whether to include fetch metadata (e.g., last fetch time) in the results.",
    )
    operators: bool | None = Field(
        default=None,
        description="Whether to apply search operators (e.g., site:example.com).",
    )


class LocalPOIsParams(BaseModel):
    """Parameters for Brave Local POIs endpoint."""

    ids: list[str] = Field(
        description="List of POI IDs to retrieve. Maximum of 20. IDs are valid for 8 hours.",
        min_length=1,
        max_length=20,
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    ui_lang: str | None = Field(
        default=None,
        description="Language code for the user interface (e.g., 'en-US', 'es-AR').",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
    )
    units: Units | None = Field(
        default=None,
        description="The units to use for the results. Options: metric/imperial",
    )


class LocalPOIsDescriptionParams(BaseModel):
    """Parameters for Brave Local POI Descriptions endpoint."""

    ids: list[str] = Field(
        description="List of POI IDs to retrieve. Maximum of 20. IDs are valid for 8 hours.",
        min_length=1,
        max_length=20,
    )


class ImageSearchParams(BaseModel):
    """Parameters for Brave Image Search endpoint."""

    q: str = Field(
        description="Search query to perform",
        min_length=1,
        max_length=400,
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
        pattern=r"^[A-Z]{2}$",
    )
    safesearch: Literal["off", "strict"] | None = Field(
        default=None,
        description="Filter out explicit content. Default is strict.",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return.",
        ge=1,
        le=200,
    )
    spellcheck: bool | None = Field(
        default=None,
        description="Attempt to correct spelling errors in the search query.",
    )


class VideoSearchParams(BaseModel):
    """Parameters for Brave Video Search endpoint."""

    q: str = Field(
        description="Search query to perform",
        min_length=1,
        max_length=400,
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    ui_lang: str | None = Field(
        default=None,
        description="Language code for the user interface (e.g., 'en-US', 'es-AR').",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
        pattern=r"^[A-Z]{2}$",
    )
    safesearch: SafeSearch | None = Field(
        default=None,
        description="Filter out explicit content. Options: off/moderate/strict",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return.",
        ge=1,
        le=50,
    )
    offset: int | None = Field(
        default=None,
        description="Skip the first N result sets/pages. Max is 9.",
        ge=0,
        le=9,
    )
    spellcheck: bool | None = Field(
        default=None,
        description="Attempt to correct spelling errors in the search query.",
    )
    freshness: Freshness | None = Field(
        default=None,
        description="Enforce freshness of results. Options: pd/pw/pm/py, or YYYY-MM-DDtoYYYY-MM-DD",
    )
    include_fetch_metadata: bool | None = Field(
        default=None,
        description="Whether to include fetch metadata (e.g., last fetch time) in the results.",
    )
    operators: bool | None = Field(
        default=None,
        description="Whether to apply search operators (e.g., site:example.com).",
    )


class NewsSearchParams(BaseModel):
    """Parameters for Brave News Search endpoint."""

    q: str = Field(
        description="Search query to perform",
        min_length=1,
        max_length=400,
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
        pattern=r"^[a-z]{2}$",
    )
    ui_lang: str | None = Field(
        default=None,
        description="Language code for the user interface (e.g., 'en-US', 'es-AR').",
        pattern=r"^[a-z]{2}-[A-Z]{2}$",
    )
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
        pattern=r"^[A-Z]{2}$",
    )
    safesearch: Literal["off", "moderate", "strict"] | None = Field(
        default=None,
        description="Filter out explicit content. Options: off/moderate/strict",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return.",
        ge=1,
        le=50,
    )
    offset: int | None = Field(
        default=None,
        description="Skip the first N result sets/pages. Max is 9.",
        ge=0,
        le=9,
    )
    spellcheck: bool | None = Field(
        default=None,
        description="Attempt to correct spelling errors in the search query.",
    )
    freshness: Freshness | None = Field(
        default=None,
        description="Enforce freshness of results. Options: pd/pw/pm/py, or YYYY-MM-DDtoYYYY-MM-DD",
    )
    extra_snippets: bool | None = Field(
        default=None,
        description="Include up to 5 text snippets for each page if possible.",
    )
    goggles: str | list[str] | None = Field(
        default=None,
        description="Goggles act as a custom re-ranking mechanism. Goggle source or URLs.",
    )
    include_fetch_metadata: bool | None = Field(
        default=None,
        description="Whether to include fetch metadata in the results.",
    )
    operators: bool | None = Field(
        default=None,
        description="Whether to apply search operators (e.g., site:example.com).",
    )


class BaseSearchHeaders(BaseModel):
    """Common headers for Brave Search endpoints."""

    x_subscription_token: str = Field(
        alias="x-subscription-token",
        description="API key for Brave Search",
    )
    api_version: str | None = Field(
        alias="api-version",
        default=None,
        description="API version to use. Default is latest available.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
    )
    accept: Literal["application/json"] | Literal["*/*"] | None = Field(
        default=None,
        description="Accept header for the request.",
    )
    cache_control: Literal["no-cache"] | None = Field(
        alias="cache-control",
        default=None,
        description="Cache control header for the request.",
    )
    user_agent: str | None = Field(
        alias="user-agent",
        default=None,
        description="User agent for the request.",
    )


class LLMContextHeaders(BaseSearchHeaders):
    """Headers for Brave LLM Context endpoint."""

    x_loc_lat: float | None = Field(
        alias="x-loc-lat",
        default=None,
        description="Latitude of the user's location.",
        ge=-90.0,
        le=90.0,
    )
    x_loc_long: float | None = Field(
        alias="x-loc-long",
        default=None,
        description="Longitude of the user's location.",
        ge=-180.0,
        le=180.0,
    )
    x_loc_city: str | None = Field(
        alias="x-loc-city",
        default=None,
        description="City of the user's location.",
    )
    x_loc_state: str | None = Field(
        alias="x-loc-state",
        default=None,
        description="State of the user's location.",
    )
    x_loc_state_name: str | None = Field(
        alias="x-loc-state-name",
        default=None,
        description="Name of the state of the user's location.",
    )
    x_loc_country: str | None = Field(
        alias="x-loc-country",
        default=None,
        description="The ISO 3166-1 alpha-2 country code of the user's location.",
    )


class LocalPOIsHeaders(BaseSearchHeaders):
    """Headers for Brave Local POIs endpoint."""

    x_loc_lat: float | None = Field(
        alias="x-loc-lat",
        default=None,
        description="Latitude of the user's location.",
        ge=-90.0,
        le=90.0,
    )
    x_loc_long: float | None = Field(
        alias="x-loc-long",
        default=None,
        description="Longitude of the user's location.",
        ge=-180.0,
        le=180.0,
    )


class LocalPOIsDescriptionHeaders(BaseSearchHeaders):
    """Headers for Brave Local POI Descriptions endpoint."""


class VideoSearchHeaders(BaseSearchHeaders):
    """Headers for Brave Video Search endpoint."""


class ImageSearchHeaders(BaseSearchHeaders):
    """Headers for Brave Image Search endpoint."""


class NewsSearchHeaders(BaseSearchHeaders):
    """Headers for Brave News Search endpoint."""


class WebSearchHeaders(BaseSearchHeaders):
    """Headers for Brave Web Search endpoint."""

    x_loc_lat: float | None = Field(
        alias="x-loc-lat",
        default=None,
        description="Latitude of the user's location.",
        ge=-90.0,
        le=90.0,
    )
    x_loc_long: float | None = Field(
        alias="x-loc-long",
        default=None,
        description="Longitude of the user's location.",
        ge=-180.0,
        le=180.0,
    )
    x_loc_timezone: str | None = Field(
        alias="x-loc-timezone",
        default=None,
        description="Timezone of the user's location.",
    )
    x_loc_city: str | None = Field(
        alias="x-loc-city",
        default=None,
        description="City of the user's location.",
    )
    x_loc_state: str | None = Field(
        alias="x-loc-state",
        default=None,
        description="State of the user's location.",
    )
    x_loc_state_name: str | None = Field(
        alias="x-loc-state-name",
        default=None,
        description="Name of the state of the user's location.",
    )
    x_loc_country: str | None = Field(
        alias="x-loc-country",
        default=None,
        description="The ISO 3166-1 alpha-2 country code of the user's location.",
    )
    x_loc_postal_code: str | None = Field(
        alias="x-loc-postal-code",
        default=None,
        description="The postal code of the user's location.",
    )

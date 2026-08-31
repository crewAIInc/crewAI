"""Pydantic input schemas for MrScraper tools."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TwoLetterCode = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z]{2}$")
]
StrictInt = Annotated[int, Field(strict=True)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

ScrapingMode = Literal["Super", "Cheap"]
StructuredDataCategory = Literal[
    "article",
    "forumThread",
    "hotel",
    "jobPosting",
    "post",
    "product",
    "property",
    "restaurant",
    "socialMediaProfile",
    "tourAttraction",
]


class MrScraperInput(BaseModel):
    """Base schema that rejects undocumented arguments."""

    model_config = ConfigDict(extra="forbid")


class GetAccountInfoInput(MrScraperInput):
    """The account operation has no model-supplied inputs."""


class MapScraperInput(MrScraperInput):
    """Inputs shared by immediate and reusable website crawl tools."""

    url: NonBlankStr = Field(description="Required starting URL to crawl.")
    max_depth: StrictInt = Field(
        default=2, description="Maximum link depth to crawl; defaults to 2."
    )
    max_pages: StrictInt = Field(
        default=50, description="Maximum pages to evaluate; defaults to 50."
    )
    limit: PositiveInt = Field(
        default=50, description="Maximum URLs to return; defaults to 50; minimum 1."
    )
    include_patterns: NonBlankStr | None = Field(
        default=None,
        description="Optional pipe-separated regular expressions for URLs to include.",
    )
    exclude_patterns: NonBlankStr | None = Field(
        default=None,
        description="Optional pipe-separated regular expressions for URLs to exclude.",
    )


class SearchGoogleSerpInput(MrScraperInput):
    """Inputs for synchronous Google SERP search."""

    query: NonBlankStr = Field(description="Required Google search query.")
    region: TwoLetterCode = Field(
        default="us", description="Two-letter result region code; defaults to 'us'."
    )
    language: TwoLetterCode = Field(
        default="en", description="Two-letter result language code; defaults to 'en'."
    )
    page: PositiveInt = Field(
        default=1, description="Google result page number; defaults to 1; minimum 1."
    )
    format: Literal["json", "html"] = Field(
        default="json",
        description="Response format: 'json' or 'html'; defaults to 'json'.",
    )
    render_js: bool = Field(
        default=False,
        description="Whether to render JavaScript before collecting results; defaults to false.",
    )


class GeneralScraperInput(MrScraperInput):
    """Inputs shared by immediate and reusable prompt scrapers."""

    url: NonBlankStr = Field(description="Required page URL to scrape.")
    prompt: NonBlankStr | None = Field(
        default=None, description="Optional extraction instructions for the AI scraper."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON object describing the expected output shape.",
    )
    mode: ScrapingMode = Field(
        default="Super",
        description="Scraping mode, 'Super' or 'Cheap'; defaults to 'Super'.",
    )
    proxy_country: TwoLetterCode | None = Field(
        default=None, description="Optional ISO country code for the proxy."
    )


class ListingScraperInput(MrScraperInput):
    """Inputs shared by immediate and reusable listing scrapers."""

    url: NonBlankStr = Field(description="Required listing page URL to scrape.")
    prompt: NonBlankStr | None = Field(
        default=None,
        description="Optional instructions describing each listing item to extract.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON object describing each expected listing item.",
    )
    max_pages: PositiveInt = Field(
        default=1,
        description="Maximum pagination pages to scrape; defaults to 1; minimum 1.",
    )
    proxy_country: TwoLetterCode | None = Field(
        default=None, description="Optional ISO country code for the proxy."
    )


class ExtractStructuredDataInput(MrScraperInput):
    """Inputs for preset structured-data extraction."""

    url: NonBlankStr = Field(description="Required page URL to scrape.")
    category: StructuredDataCategory = Field(
        default="article",
        description="Structured extraction preset category; defaults to 'article'.",
    )
    mode: ScrapingMode = Field(
        default="Super",
        description="Scraping mode, 'Super' or 'Cheap'; defaults to 'Super'.",
    )
    proxy_country: TwoLetterCode | None = Field(
        default=None, description="Optional ISO country code for the proxy."
    )


class FetchRenderedHtmlInput(MrScraperInput):
    """Inputs for the rendered-page API."""

    url: NonBlankStr = Field(description="Required target URL to render.")
    max_retries: NonNegativeInt = Field(
        default=3, description="Maximum retry attempts; defaults to 3; minimum 0."
    )
    timeout: PositiveInt = Field(
        default=300,
        description="Maximum page-load time in seconds; defaults to 300; minimum 1.",
    )
    geo_code: TwoLetterCode = Field(
        default="us", description="Geolocation country code; defaults to 'us'."
    )
    proxy_country: TwoLetterCode = Field(
        default="us", description="Proxy country code; defaults to 'us'."
    )
    screenshot: bool = Field(
        default=False, description="Whether to capture a screenshot; defaults to false."
    )
    screenshot_mode: Literal["full", "top"] | None = Field(
        default=None,
        description="Optional screenshot mode; used only when screenshot is true.",
    )
    html: bool = Field(
        default=True, description="Whether to include rendered HTML; defaults to true."
    )
    markdown: bool = Field(
        default=False, description="Whether to include Markdown; defaults to false."
    )
    token_cap: PositiveInt | None = Field(
        default=None,
        description="Optional maximum processing token allowance; minimum 1.",
    )
    wait_for_selector: NonBlankStr | None = Field(
        default=None, description="Optional CSS selector to await before returning."
    )
    wait_until: Literal["domcontentloaded", "load", "networkidle"] | None = Field(
        default=None,
        description="Optional browser lifecycle event to await.",
    )
    block_resources: bool = Field(
        default=False,
        description="Whether to block images, fonts, and stylesheets; defaults to false.",
    )
    home_page: bool = Field(
        default=False,
        description="Whether to visit the site home page first; defaults to false.",
    )
    return_cookie: bool = Field(
        default=False,
        description="Whether to include browser cookies; defaults to false.",
    )
    super_mode: bool = Field(
        default=False,
        description="Whether to use stronger device mode; defaults to false.",
    )


class GetResultsInput(MrScraperInput):
    """Inputs for paginated scraper results."""

    scraper_id: NonBlankStr = Field(
        description="Required scraper ID whose results to list."
    )
    page: StrictInt = Field(
        default=1, description="Results page number; defaults to 1."
    )
    page_size: StrictInt = Field(
        default=10, description="Number of results per page; defaults to 10."
    )
    sort_by: Literal["createdAt"] = Field(
        default="createdAt", description="Sort field; only 'createdAt' is supported."
    )
    sort_order: Literal["ASC", "DESC"] = Field(
        default="DESC",
        description="Sort direction, 'ASC' or 'DESC'; defaults to 'DESC'.",
    )


class GetLatestResultsInput(MrScraperInput):
    """Inputs for the newest scraper results."""

    scraper_id: NonBlankStr = Field(
        description="Required scraper ID whose newest results to list."
    )
    count: StrictInt = Field(
        default=10, description="Number of newest results; defaults to 10."
    )


class GetResultDetailInput(MrScraperInput):
    """Inputs for one result record."""

    result_id: NonBlankStr = Field(description="Required result ID to retrieve.")


class RunExistingScraperInput(MrScraperInput):
    """Stable conditional schema for AI and manual single scraper runs."""

    scraper_type: Literal["ai", "manual"] = Field(
        description="Required scraper kind selecting the AI or manual endpoint."
    )
    scraper_id: NonBlankStr = Field(description="Required existing scraper ID.")
    url: NonBlankStr = Field(description="Required URL to process in this run.")
    max_retry: NonNegativeInt = Field(
        default=3, description="Maximum retry attempts; defaults to 3; minimum 0."
    )
    proxy_country: TwoLetterCode | None = Field(
        default=None, description="Optional proxy country code."
    )
    agent_type: Literal["general", "listing", "map"] = Field(
        default="general",
        description="AI agent type; defaults to 'general' for AI and is forbidden for manual runs.",
    )

    bypass_proxy: bool | None = Field(
        default=None,
        description="General/Listing default false; Manual default true; forbidden for Map.",
    )
    html: bool | None = Field(
        default=None,
        description="Optional General, Listing, or Manual HTML output flag.",
    )
    markdown: bool | None = Field(
        default=None,
        description="Optional General, Listing, or Manual Markdown output flag.",
    )
    screenshot: bool | None = Field(
        default=None,
        description="Optional General, Listing, or Manual screenshot flag.",
    )
    stream: bool | None = Field(
        default=None,
        description="Optional Listing or Manual streaming flag.",
    )
    timeout: PositiveInt | None = Field(
        default=None,
        description="Listing timeout defaults to 300; Manual timeout defaults to 600; minimum 1.",
    )

    render_javascript: bool | None = Field(
        default=None,
        description="Optional General/Listing JavaScript rendering flag.",
    )
    return_cookies: bool | None = Field(
        default=None,
        description="Optional General/Listing cookie-return flag.",
    )
    use_home_page: bool | None = Field(
        default=None,
        description="Optional General/Listing home-page visit flag.",
    )
    wait_for_selector: NonBlankStr | None = Field(
        default=None, description="Optional General/Listing CSS selector to await."
    )

    max_pages: PositiveInt | None = Field(
        default=None,
        description="Listing defaults to 5; Map defaults to 50; minimum 1.",
    )

    max_depth: NonNegativeInt | None = Field(
        default=None, description="Optional Map crawl depth; minimum 0."
    )
    limit: PositiveInt | None = Field(
        default=None, description="Optional Map result limit; minimum 1."
    )
    include_patterns: NonBlankStr | None = Field(
        default=None, description="Optional Map include-pattern expressions."
    )
    exclude_patterns: NonBlankStr | None = Field(
        default=None, description="Optional Map exclude-pattern expressions."
    )

    cookie_jar: NonBlankStr | None = Field(
        default=None, description="Optional Manual cookie-jar identifier or value."
    )
    cookies: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional Manual browser-cookie objects.",
    )
    home_page: bool | None = Field(
        default=None, description="Optional Manual home-page visit flag."
    )
    home_page_timeout: PositiveInt | None = Field(
        default=None, description="Optional Manual home-page timeout; minimum 1."
    )
    paginator: dict[str, Any] | None = Field(
        default=None,
        description="Optional Manual paginator configuration.",
    )
    proxy: NonBlankStr | None = Field(
        default=None, description="Optional Manual proxy URL."
    )
    record: bool | None = Field(
        default=None,
        description="Optional Manual browser-session recording flag.",
    )
    return_cookie: bool | None = Field(
        default=None, description="Optional Manual cookie-return flag."
    )
    token_cap: NonNegativeInt | None = Field(
        default=None, description="Optional Manual token cap; minimum 0."
    )

    @model_validator(mode="after")
    def validate_conditional_fields(self) -> RunExistingScraperInput:
        """Reject explicitly supplied fields that do not apply to the selected run."""
        supplied = self.model_fields_set
        common = {"scraper_type", "scraper_id", "url", "max_retry", "proxy_country"}
        general = {
            "agent_type",
            "bypass_proxy",
            "html",
            "markdown",
            "render_javascript",
            "return_cookies",
            "screenshot",
            "use_home_page",
            "wait_for_selector",
        }
        listing = general | {"max_pages", "timeout", "stream"}
        mapping = {
            "agent_type",
            "max_depth",
            "max_pages",
            "limit",
            "include_patterns",
            "exclude_patterns",
        }
        manual = {
            "bypass_proxy",
            "cookie_jar",
            "cookies",
            "home_page",
            "home_page_timeout",
            "html",
            "markdown",
            "paginator",
            "proxy",
            "record",
            "return_cookie",
            "screenshot",
            "stream",
            "timeout",
            "token_cap",
        }

        if self.scraper_type == "manual":
            incompatible = supplied - common - manual
            if incompatible:
                names = ", ".join(sorted(incompatible))
                raise ValueError(f"Manual scraper runs do not accept: {names}")
            return self

        allowed = {
            "general": general,
            "listing": listing,
            "map": mapping,
        }[self.agent_type]
        incompatible = supplied - common - allowed
        if incompatible:
            names = ", ".join(sorted(incompatible))
            raise ValueError(
                f"AI {self.agent_type} scraper runs do not accept: {names}"
            )
        return self


class RunExistingScraperBatchInput(MrScraperInput):
    """Inputs for batch runs of an existing scraper."""

    scraper_type: Literal["ai", "manual"] = Field(
        description="Required scraper kind selecting the AI or manual bulk endpoint."
    )
    scraper_id: NonBlankStr = Field(description="Required existing scraper ID.")
    urls: list[NonBlankStr] = Field(
        min_length=1,
        description="Required nonempty array of nonblank URLs to process in this batch.",
    )

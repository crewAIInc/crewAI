"""Input schemas for TinyFish tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TinyfishAgentParams(BaseModel):
    """Input schema for a synchronous TinyFish browser automation."""

    url: str = Field(..., description="Target website URL to automate.")
    goal: str = Field(
        ...,
        description=(
            "Natural language description of what to accomplish on the website "
            "(e.g., 'extract product names and prices', 'fill the contact form')."
        ),
    )
    browser_profile: Literal["lite", "stealth"] = Field(
        default="lite",
        description=(
            'Browser execution mode. "lite" is fast for standard sites; '
            '"stealth" enables anti-detection for sites with bot protection.'
        ),
    )

    @field_validator("url")
    @classmethod
    def _check_url_scheme(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class TinyfishSearchParams(BaseModel):
    """Input schema for the search endpoint."""

    query: str = Field(..., description="Search query string.")
    location: str | None = Field(
        default=None,
        description="Optional location to scope results, e.g. 'United States'.",
    )
    language: str | None = Field(
        default=None,
        description="Optional language code, e.g. 'en'.",
    )


class TinyfishFetchParams(BaseModel):
    """Input schema for the fetch endpoint."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="One to ten URLs to fetch and extract clean content from.",
    )
    format: Literal["markdown", "html", "json"] = Field(
        default="markdown",
        description="Output format for extracted content.",
    )
    links: bool = Field(default=False, description="Whether to include page links.")
    image_links: bool = Field(
        default=False,
        description="Whether to include image links.",
    )

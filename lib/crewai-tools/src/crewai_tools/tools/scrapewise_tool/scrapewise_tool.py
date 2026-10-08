from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_SAMPLE_ROWS = 100
"""The ScrapeWise API caps sample data at 100 rows."""

INTERNAL_COLUMN_PREFIX = "_sw_"
"""Prefix on platform bookkeeping columns, which are noise to a model."""


class ScrapewiseError(Exception):
    """Base exception for ScrapeWise-related errors."""


class FixedScrapewiseProductDataToolSchema(BaseModel):
    """Input for ScrapewiseProductDataTool when scraper_id is fixed."""

    max_rows: int = Field(
        default=25,
        description="Maximum number of product rows to return, between 1 and 100",
    )

    @field_validator("max_rows")
    @classmethod
    def validate_max_rows(cls, v: int) -> int:
        """Validate the row cap."""
        if not 1 <= v <= MAX_SAMPLE_ROWS:
            raise ValueError(f"max_rows must be between 1 and {MAX_SAMPLE_ROWS}")
        return v


class ScrapewiseProductDataToolSchema(FixedScrapewiseProductDataToolSchema):
    """Input for ScrapewiseProductDataTool."""

    scraper_id: str = Field(
        ...,
        description="Mandatory id of the ScrapeWise scraper to read product data from",
    )


class ScrapewiseProductDataTool(BaseTool):
    """A tool that reads structured competitor product and pricing data from ScrapeWise.

    ScrapeWise runs managed scrapers against e-commerce sites and stores the
    extracted rows -- title, price, availability, product identifiers. This
    tool reads those stored rows, so the agent gets structured data without
    fetching or parsing any HTML.

    Construct with a `scraper_id` to pin the tool to one site, or leave it out
    and let the agent pass the id per call. Scraper ids are listed in the
    ScrapeWise dashboard at https://portal.scrapewise.ai.

    Raises:
        ValueError: If API key is missing or max_rows is out of range
        ScrapewiseError: If the ScrapeWise API returns an error
        RuntimeError: If the read operation fails
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ScrapeWise product data reader"
    description: str = (
        "A tool that reads structured competitor product and pricing data "
        "collected by ScrapeWise scrapers."
    )
    args_schema: type[BaseModel] = ScrapewiseProductDataToolSchema
    scraper_id: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    _client: Any = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["scrapewise"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SCRAPEWISE_API_KEY",
                description="API key for the ScrapeWise API",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        scraper_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            from scrapewise import ScrapewiseClient

        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'scrapewise' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "scrapewise"], check=True)  # noqa: S607
                from scrapewise import ScrapewiseClient

            else:
                raise ImportError(
                    "`scrapewise` package not found, please run `uv add scrapewise`"
                ) from None

        self.api_key = api_key or os.getenv("SCRAPEWISE_API_KEY")

        if not self.api_key:
            raise ValueError("ScrapeWise API key is required")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url is not None:
            self.base_url = base_url
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            self.timeout = timeout
            client_kwargs["timeout"] = timeout

        self._client = ScrapewiseClient(**client_kwargs)

        if scraper_id is not None:
            self.scraper_id = scraper_id
            self.description = (
                "A tool that reads structured competitor product and pricing "
                f"data collected by ScrapeWise scraper {scraper_id}."
            )
            self.args_schema = FixedScrapewiseProductDataToolSchema

    def _format_rows(self, rows: Any, max_rows: int) -> str:
        """Handle and validate the ScrapeWise response."""
        if rows is None:
            raise RuntimeError("Empty response from ScrapeWise API")

        if isinstance(rows, dict):
            rows = rows.get("content") or rows.get("items") or []

        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected response from ScrapeWise API: {type(rows).__name__}"
            )

        if not rows:
            return (
                "No product rows stored for this scraper yet. The scraper may "
                "never have run, or its last run may have failed."
            )

        lines = []
        for row in rows[:max_rows]:
            if not isinstance(row, dict):
                lines.append(str(row))
                continue
            lines.append(
                " | ".join(
                    f"{key}: {value}"
                    for key, value in row.items()
                    if not key.startswith(INTERNAL_COLUMN_PREFIX)
                    and value not in (None, "")
                )
            )

        header = f"{min(len(rows), max_rows)} product row(s) from ScrapeWise:"
        return "\n".join([header, *lines])

    def _run(self, **kwargs: Any) -> str:
        scraper_id = kwargs.get("scraper_id") or self.scraper_id
        max_rows = kwargs.get("max_rows", 25)

        if not scraper_id:
            return (
                "Error: no scraper_id given. Pass one, or construct "
                "ScrapewiseProductDataTool with a fixed scraper_id."
            )

        try:
            rows = self._client.get_sample_data(scraper_id)
            return self._format_rows(rows, max_rows)
        except Exception as e:
            return f"Error reading ScrapeWise product data: {e}"

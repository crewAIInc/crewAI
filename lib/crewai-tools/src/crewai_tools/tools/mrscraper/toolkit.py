"""Convenience factory for independent MrScraper tool instances."""

from collections.abc import Iterable

from crewai.tools import BaseTool

from crewai_tools.tools.mrscraper.account import MrScraperGetAccountInfoTool
from crewai_tools.tools.mrscraper.base import MrScraperBaseTool, resolve_api_token
from crewai_tools.tools.mrscraper.client import MrScraperClient
from crewai_tools.tools.mrscraper.discovery import (
    MrScraperCrawlWebsiteUrlsTool,
    MrScraperSearchGoogleSerpTool,
)
from crewai_tools.tools.mrscraper.extraction import (
    MrScraperExtractListingsTool,
    MrScraperExtractPageByPromptTool,
    MrScraperExtractStructuredDataTool,
    MrScraperFetchRenderedHtmlTool,
)
from crewai_tools.tools.mrscraper.results import (
    MrScraperGetLatestResultsTool,
    MrScraperGetResultDetailTool,
    MrScraperGetResultsTool,
)
from crewai_tools.tools.mrscraper.scraper_creation import (
    MrScraperCreateListingScraperTool,
    MrScraperCreatePromptScraperTool,
    MrScraperCreateWebsiteCrawlScraperTool,
)
from crewai_tools.tools.mrscraper.scraper_runs import (
    MrScraperRunExistingScraperBatchTool,
    MrScraperRunExistingScraperTool,
)


ToolClass = type[MrScraperBaseTool]

_GROUPS: dict[str, tuple[ToolClass, ...]] = {
    "account": (MrScraperGetAccountInfoTool,),
    "discovery": (
        MrScraperCrawlWebsiteUrlsTool,
        MrScraperSearchGoogleSerpTool,
    ),
    "extraction": (
        MrScraperExtractPageByPromptTool,
        MrScraperExtractListingsTool,
        MrScraperExtractStructuredDataTool,
        MrScraperFetchRenderedHtmlTool,
    ),
    "results": (
        MrScraperGetResultsTool,
        MrScraperGetLatestResultsTool,
        MrScraperGetResultDetailTool,
    ),
    "scraper creation": (
        MrScraperCreatePromptScraperTool,
        MrScraperCreateListingScraperTool,
        MrScraperCreateWebsiteCrawlScraperTool,
    ),
    "scraper runs": (
        MrScraperRunExistingScraperTool,
        MrScraperRunExistingScraperBatchTool,
    ),
}


def create_mrscraper_toolkit(
    *,
    groups: Iterable[str] | None = None,
    tool_names: Iterable[str] | None = None,
    api_token: str | None = None,
) -> list[BaseTool]:
    """Create all 15 MrScraper tools or a selected group/name subset.

    Args:
        groups: Optional case-insensitive group names: Account, Discovery,
            Extraction, Results, Scraper Creation, or Scraper Runs.
        tool_names: Optional exact public tool names to select.
        api_token: Optional constructor-only credential override. The value remains
            private and is excluded from schemas, serialization, reprs, and errors.

    Returns:
        New independent tool instances sharing one configured HTTP client.

    Raises:
        ValueError: If selection is ambiguous or contains an unknown group/name.
    """
    if groups is not None and tool_names is not None:
        raise ValueError("Select MrScraper tools by groups or tool_names, not both")

    all_classes = tuple(tool for group in _GROUPS.values() for tool in group)
    selected: tuple[ToolClass, ...]
    if groups is not None:
        normalized = list(dict.fromkeys(group.strip().lower() for group in groups))
        unknown = sorted(set(normalized) - _GROUPS.keys())
        if unknown:
            raise ValueError(f"Unknown MrScraper toolkit groups: {', '.join(unknown)}")
        selected = tuple(tool for group in normalized for tool in _GROUPS[group])
    elif tool_names is not None:
        by_name = {tool.model_fields["name"].default: tool for tool in all_classes}
        requested = list(tool_names)
        unknown = sorted(set(requested) - by_name.keys())
        if unknown:
            raise ValueError(f"Unknown MrScraper tool names: {', '.join(unknown)}")
        selected = tuple(by_name[name] for name in requested)
    else:
        selected = all_classes

    client = MrScraperClient(resolve_api_token(api_token))
    return [tool_class(client=client) for tool_class in selected]


__all__ = ["create_mrscraper_toolkit"]

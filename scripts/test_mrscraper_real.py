"""Run opt-in, real API smoke tests for every MrScraper CrewAI tool.

Examples:
    uv run python scripts/test_mrscraper_real.py --list
    uv run python scripts/test_mrscraper_real.py --test account
    uv run python scripts/test_mrscraper_real.py --test rendered_html
    uv run python scripts/test_mrscraper_real.py --test crew_agent

These tests call real services and may consume MrScraper or LLM credits. Run one
test at a time. Scraper creation tests create persistent scraper records.
"""

# ruff: noqa: T201 - This is an intentionally interactive command-line script.

from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from typing import Any

from crewai import Agent, Crew, Task
from crewai_tools import (
    MrScraperCrawlWebsiteUrlsTool,
    MrScraperCreateListingScraperTool,
    MrScraperCreatePromptScraperTool,
    MrScraperCreateWebsiteCrawlScraperTool,
    MrScraperExtractListingsTool,
    MrScraperExtractPageByPromptTool,
    MrScraperExtractStructuredDataTool,
    MrScraperFetchRenderedHtmlTool,
    MrScraperGetAccountInfoTool,
    MrScraperGetLatestResultsTool,
    MrScraperGetResultDetailTool,
    MrScraperGetResultsTool,
    MrScraperRunExistingScraperBatchTool,
    MrScraperRunExistingScraperTool,
    MrScraperSearchGoogleSerpTool,
)


# Read credentials from the environment so they cannot be committed accidentally.
MRSCRAPER_API_TOKEN = os.getenv("MRSCRAPER_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Safe public targets for a first smoke test. Replace them as needed.
TARGET_URL = "https://www.cireba.com/property-detail/south-sound/residential-properties-for-sale-in-cayman-islands/castillo-caribe-3"
LISTING_URL = "https://www.cireba.com/cayman-islands-real-estate-listings/"
SEARCH_QUERY = "CrewAI framework"

# Fill these IDs before running result/rerun tests.
AI_SCRAPER_ID = "0bc62b79-e314-4d70-a6c8-7f0bd58ae221"
MANUAL_SCRAPER_ID = ""
RESULT_ID = ""

MAX_OUTPUT_CHARS = 6_000
Test = Callable[[], Any]


def tool(tool_class: type[Any]) -> Any:
    """Construct a tool with the configured token."""
    require(MRSCRAPER_API_TOKEN, "MRSCRAPER_API_TOKEN")
    return tool_class(api_token=MRSCRAPER_API_TOKEN)


def require(value: str, name: str) -> str:
    """Require a configured value without printing secret contents."""
    if not value.strip():
        raise RuntimeError(
            f"{name} belum diisi. Set environment variable atau isi konstanta "
            "di bagian atas file ini."
        )
    return value


def account() -> str:
    return tool(MrScraperGetAccountInfoTool).run()


def crawl_urls() -> str:
    return tool(MrScraperCrawlWebsiteUrlsTool).run(
        url=TARGET_URL, max_depth=1, max_pages=2, limit=5
    )


def google_serp() -> str:
    return tool(MrScraperSearchGoogleSerpTool).run(
        query=SEARCH_QUERY, region="us", language="en", page=1, format="json"
    )


def extract_prompt() -> str:
    return tool(MrScraperExtractPageByPromptTool).run(
        url=TARGET_URL,
        prompt="Extract the page title and main description.",
        output_schema={"title": "string", "description": "string"},
        mode="Cheap",
    )


def extract_listings() -> str:
    return tool(MrScraperExtractListingsTool).run(
        url=LISTING_URL,
        prompt="Extract book title and price from the first page.",
        output_schema={"title": "string", "price": "string"},
        max_pages=1,
    )


def extract_structured() -> str:
    return tool(MrScraperExtractStructuredDataTool).run(
        url=TARGET_URL, category="article", mode="Cheap"
    )


def rendered_html() -> str:
    # Advanced options that remain False/None are intentionally not sent.
    return tool(MrScraperFetchRenderedHtmlTool).run(
        url=TARGET_URL,
        html=True,
        home_page=True,
        markdown=False,
        screenshot=False,
        wait_until=None,
        wait_for_selector=None,
    )


def get_results() -> str:
    return tool(MrScraperGetResultsTool).run(
        scraper_id=require(AI_SCRAPER_ID, "AI_SCRAPER_ID"),
        page=1,
        page_size=5,
        sort_order="DESC",
    )


def get_latest_results() -> str:
    return tool(MrScraperGetLatestResultsTool).run(
        scraper_id=require(AI_SCRAPER_ID, "AI_SCRAPER_ID"), count=5
    )


def get_result_detail() -> str:
    return tool(MrScraperGetResultDetailTool).run(
        result_id=require(RESULT_ID, "RESULT_ID")
    )


def create_prompt_scraper() -> str:
    return tool(MrScraperCreatePromptScraperTool).run(
        url=TARGET_URL,
        prompt="Extract the property name and price, number of bedroom and bathroom, and mls ID.",
        output_schema={"title": "string", "description": "string"},
        mode="Cheap",
    )


def create_listing_scraper() -> str:
    return tool(MrScraperCreateListingScraperTool).run(
        url=LISTING_URL,
        prompt="Extract book title and price.",
        output_schema={"title": "string", "price": "string"},
        max_pages=1,
    )


def create_crawl_scraper() -> str:
    return tool(MrScraperCreateWebsiteCrawlScraperTool).run(
        url=TARGET_URL, max_depth=1, max_pages=2, limit=5
    )


def run_ai_scraper() -> str:
    return tool(MrScraperRunExistingScraperTool).run(
        scraper_type="ai",
        scraper_id=require(AI_SCRAPER_ID, "AI_SCRAPER_ID"),
        url=TARGET_URL,
        agent_type="general",
    )


def run_manual_scraper() -> str:
    return tool(MrScraperRunExistingScraperTool).run(
        scraper_type="manual",
        scraper_id=require(MANUAL_SCRAPER_ID, "MANUAL_SCRAPER_ID"),
        url=TARGET_URL,
    )


def run_ai_batch() -> str:
    return tool(MrScraperRunExistingScraperBatchTool).run(
        scraper_type="ai",
        scraper_id=require(AI_SCRAPER_ID, "AI_SCRAPER_ID"),
        urls=[TARGET_URL, f"{TARGET_URL}/?second=1"],
    )


def crew_agent() -> Any:
    """Test MrScraper through a real CrewAI Agent and OpenAI model."""
    require(OPENAI_API_KEY, "OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    fetch_tool = tool(MrScraperFetchRenderedHtmlTool)
    researcher = Agent(
        role="Web content tester",
        goal="Fetch one public page and report its title accurately",
        backstory="You test web extraction tools with narrow, low-cost calls.",
        tools=[fetch_tool],
        verbose=True,
    )
    task = Task(
        description=(
            f"Use the MrScraper rendered HTML tool to fetch {TARGET_URL}. "
            "Return the page title and a one-sentence summary."
        ),
        expected_output="The source URL, page title, and one-sentence summary.",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], verbose=True).kickoff()


TESTS: dict[str, Test] = {
    "account": account,
    "crawl_urls": crawl_urls,
    "google_serp": google_serp,
    "extract_prompt": extract_prompt,
    "extract_listings": extract_listings,
    "extract_structured": extract_structured,
    "rendered_html": rendered_html,
    "get_results": get_results,
    "get_latest_results": get_latest_results,
    "get_result_detail": get_result_detail,
    "create_prompt_scraper": create_prompt_scraper,
    "create_listing_scraper": create_listing_scraper,
    "create_crawl_scraper": create_crawl_scraper,
    "run_ai_scraper": run_ai_scraper,
    "run_manual_scraper": run_manual_scraper,
    "run_ai_batch": run_ai_batch,
    "crew_agent": crew_agent,
}

CREATES_RECORDS = {
    "create_prompt_scraper",
    "create_listing_scraper",
    "create_crawl_scraper",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--test", choices=sorted(TESTS), help="Run one real test")
    args = parser.parse_args()

    if args.list or args.test is None:
        print("Available real tests:")
        for name in TESTS:
            warning = " [CREATES A SCRAPER]" if name in CREATES_RECORDS else ""
            print(f"  {name}{warning}")
        if args.test is None:
            print("\nRun one with: --test <name>")
        return 0

    print(f"Running real test: {args.test}")
    try:
        result = TESTS[args.test]()
    except Exception as exc:  # This CLI should show upstream integration failures.
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    output = str(result)
    if len(output) > MAX_OUTPUT_CHARS:
        output = f"{output[:MAX_OUTPUT_CHARS]}\n... [output truncated]"
    print("SUCCESS")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

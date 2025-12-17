"""Olostep Tool for CrewAI - Web scraping, crawling, mapping, and AI search.

Olostep (https://olostep.com) provides:
- Scraping: Extract clean content from any webpage
- Crawling: Follow links and extract content from multiple pages
- Sitemap: Discover all URLs on a website
- Answers: AI-powered web search with structured JSON output
- LLM Extract: Extract structured data from pages using schemas
- Parsers: Built-in parsers for Google Search, News, LinkedIn, etc.
"""

import logging
import subprocess
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


# ============================================================================
# Input Schemas for different modes
# ============================================================================

class OlostepScrapeSchema(BaseModel):
    """Input schema for scraping a single page."""

    url: str = Field(
        ..., description="The URL of the webpage to scrape"
    )
    formats: list[str] | None = Field(
        default=None,
        description="Output formats: 'markdown', 'html', 'text', 'json'. Defaults to ['markdown']"
    )
    wait_before_scraping: int | None = Field(
        default=None,
        description="Milliseconds to wait for JavaScript content to load"
    )
    country: str | None = Field(
        default=None,
        description="Country code for geolocation (e.g., 'US', 'UK', 'DE')"
    )
    parser: str | None = Field(
        default=None,
        description="Parser ID for structured extraction (e.g., '@olostep/google-search', '@olostep/linkedin-profile')"
    )
    llm_extract_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for LLM-powered structured data extraction"
    )


class OlostepCrawlSchema(BaseModel):
    """Input schema for crawling a website."""

    url: str = Field(
        ..., description="The starting URL for the crawl"
    )
    max_pages: int = Field(
        default=10,
        description="Maximum number of pages to crawl (1-1000)"
    )
    include_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to include (regex)"
    )
    exclude_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to exclude (regex)"
    )

    @field_validator("max_pages")
    @classmethod
    def validate_max_pages(cls, v: int) -> int:
        if v < 1:
            return 1
        if v > 1000:
            return 1000
        return v


class OlostepMapSchema(BaseModel):
    """Input schema for mapping/discovering URLs on a website."""

    url: str = Field(
        ..., description="The website URL to map"
    )


class OlostepAnswerSchema(BaseModel):
    """Input schema for AI-powered web search answers."""

    task: str = Field(
        ..., description="The question or task for AI to answer using web search"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema to structure the answer output"
    )


class OlostepToolSchema(BaseModel):
    """Main input schema for OlostepTool - supports all modes."""

    url: str | None = Field(
        default=None,
        description="URL to process (required for scrape, crawl, map modes)"
    )
    mode: Literal["scrape", "crawl", "map", "answer"] = Field(
        default="scrape",
        description="Operation mode: 'scrape' (single page), 'crawl' (follow links), 'map' (discover URLs), 'answer' (AI search)"
    )
    
    # Scrape-specific
    formats: list[str] | None = Field(
        default=None,
        description="Output formats for scrape mode: 'markdown', 'html', 'text', 'json'"
    )
    wait_before_scraping: int | None = Field(
        default=None,
        description="Milliseconds to wait for JavaScript content"
    )
    country: str | None = Field(
        default=None,
        description="Country code for geolocation (e.g., 'US', 'UK')"
    )
    parser: str | None = Field(
        default=None,
        description="Parser ID (e.g., '@olostep/google-search', '@olostep/linkedin-profile')"
    )
    llm_extract_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for LLM extraction"
    )
    
    # Crawl-specific
    max_pages: int | None = Field(
        default=None,
        description="Max pages for crawl mode (1-1000)"
    )
    include_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to include in crawl"
    )
    exclude_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to exclude from crawl"
    )
    
    # Answer-specific
    task: str | None = Field(
        default=None,
        description="Question/task for answer mode"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for answer output"
    )

    @field_validator("max_pages")
    @classmethod
    def validate_max_pages(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1:
            return 1
        if v > 1000:
            return 1000
        return v


# ============================================================================
# Main Tool Class
# ============================================================================

class OlostepTool(BaseTool):
    """Tool for web scraping, crawling, and AI-powered search using Olostep.

    Olostep (https://olostep.com) provides LLM-ready web content extraction
    with support for JavaScript rendering and dynamic content.

    **Modes:**
    
    - **scrape**: Extract content from a single webpage
      - Supports markdown, HTML, text, JSON formats
      - LLM extraction with JSON schemas
      - Built-in parsers for Google, LinkedIn, etc.
      - Country-based geolocation
      
    - **crawl**: Follow links and extract content from multiple pages
      - Configurable max pages (up to 1000)
      - Include/exclude URL patterns
      
    - **map**: Discover all URLs on a website
      - Returns a comprehensive sitemap
      
    - **answer**: AI-powered web search
      - Ask questions in natural language
      - Get structured JSON responses
      - Sources included for verification

    **Example parsers:**
    - @olostep/google-search - Google search results
    - @olostep/google-news - Google News articles
    - @olostep/linkedin-profile - LinkedIn profiles
    - @olostep/google-maps - Google Maps places
    - @olostep/perplexity-search - Perplexity AI search

    Attributes:
        name: Tool name for identification
        description: Tool description for LLM understanding
        api_key: Olostep API key
        default_mode: Default operation mode
        default_formats: Default output formats
    """

    name: str = "OlostepTool"
    description: str = (
        "The most reliable and cost-effective web search, scraping and crawling API for AI. "
        "Build intelligent agents that can search, scrape, analyze, and structure data from any website.\n\n"
        "Modes: 'scrape' (single page), 'crawl' (multiple pages), 'map' (sitemap), 'answer' (AI search).\n"
        "Features: LLM extraction, parsers (@olostep/google-search, etc.), country geolocation, JS rendering."
    )
    args_schema: type[BaseModel] = OlostepToolSchema
    
    # Configuration
    api_key: str | None = None
    default_mode: Literal["scrape", "crawl", "map", "answer"] = "scrape"
    default_formats: list[str] = Field(default_factory=lambda: ["markdown"])
    default_max_pages: int = 10
    
    # Internal state
    _client: Any = None
    log_failures: bool = True
    
    package_dependencies: list[str] = Field(default_factory=lambda: ["olostep"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OLOSTEP_API_KEY",
                description="API key for Olostep. Get one at https://olostep.com/dashboard",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        default_mode: Literal["scrape", "crawl", "map", "answer"] = "scrape",
        default_formats: list[str] | None = None,
        default_max_pages: int = 10,
        log_failures: bool = True,
        **kwargs,
    ):
        """Initialize OlostepTool.

        Args:
            api_key: Olostep API key. Uses OLOSTEP_API_KEY env var if not provided.
            default_mode: Default operation mode.
            default_formats: Default output formats for scraping.
            default_max_pages: Default max pages for crawling.
            log_failures: Whether to log errors.
            **kwargs: Additional arguments for BaseTool.
        """
        super().__init__(**kwargs)
        
        self.api_key = api_key
        self.default_mode = default_mode
        self.default_formats = default_formats or ["markdown"]
        self.default_max_pages = default_max_pages
        self.log_failures = log_failures
        
        # Initialize client lazily
        self._client = None

    def _get_client(self) -> Any:
        """Get or create the Olostep client."""
        if self._client is None:
            try:
                from olostep import SyncOlostepClient
            except ImportError:
                import click

                if click.confirm(
                    "The 'olostep' package is required. Install it now?"
                ):
                    subprocess.run(["uv", "pip", "install", "olostep"], check=True)  # noqa: S607
                    from olostep import SyncOlostepClient
                else:
                    raise ImportError(
                        "olostep package not found. Install with: pip install olostep"
                    ) from None

            self._client = SyncOlostepClient(api_key=self.api_key)
        
        return self._client

    def _run(
        self,
        url: str | None = None,
        mode: Literal["scrape", "crawl", "map", "answer"] | None = None,
        # Scrape options
        formats: list[str] | None = None,
        wait_before_scraping: int | None = None,
        country: str | None = None,
        parser: str | None = None,
        llm_extract_schema: dict[str, Any] | None = None,
        # Crawl options
        max_pages: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        # Answer options
        task: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Execute the Olostep tool.

        Args:
            url: URL to process (required for scrape/crawl/map modes)
            mode: Operation mode
            formats: Output formats for scrape mode
            wait_before_scraping: Wait time in ms for JS content
            country: Country code for geolocation
            parser: Parser ID for structured extraction
            llm_extract_schema: JSON schema for LLM extraction
            max_pages: Max pages for crawl mode
            include_patterns: URL patterns to include in crawl
            exclude_patterns: URL patterns to exclude from crawl
            task: Question for answer mode
            json_schema: JSON schema for answer output

        Returns:
            Extracted content, crawl summary, URL list, or AI answer
        """
        mode = mode or self.default_mode

        try:
            if mode == "scrape":
                if not url:
                    return "Error: URL is required for scrape mode"
                return self._scrape(
                    url=url,
                    formats=formats,
                    wait_before_scraping=wait_before_scraping,
                    country=country,
                    parser=parser,
                    llm_extract_schema=llm_extract_schema,
                )
            
            elif mode == "crawl":
                if not url:
                    return "Error: URL is required for crawl mode"
                return self._crawl(
                    url=url,
                    max_pages=max_pages,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
            
            elif mode == "map":
                if not url:
                    return "Error: URL is required for map mode"
                return self._map(url=url)
            
            elif mode == "answer":
                if not task:
                    return "Error: 'task' is required for answer mode"
                return self._answer(task=task, json_schema=json_schema)
            
            else:
                return f"Error: Unknown mode '{mode}'. Use 'scrape', 'crawl', 'map', or 'answer'."

        except Exception as e:
            error_msg = f"Olostep error: {str(e)}"
            if self.log_failures:
                logger.error(error_msg)
            return error_msg

    def _scrape(
        self,
        url: str,
        formats: list[str] | None = None,
        wait_before_scraping: int | None = None,
        country: str | None = None,
        parser: str | None = None,
        llm_extract_schema: dict[str, Any] | None = None,
    ) -> str:
        """Scrape a single webpage.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, text, json)
            wait_before_scraping: Wait time in ms
            country: Country code for geolocation
            parser: Parser ID for structured extraction
            llm_extract_schema: JSON schema for LLM extraction

        Returns:
            Scraped content
        """
        client = self._get_client()
        
        # Build parameters
        kwargs: dict[str, Any] = {
            "url": url,
        }
        
        if formats:
            kwargs["formats"] = formats
        else:
            kwargs["formats"] = self.default_formats
            
        if wait_before_scraping is not None:
            kwargs["wait_before_scraping"] = wait_before_scraping
            
        if country:
            kwargs["country"] = country
            
        if parser:
            kwargs["parser"] = {"id": parser}
            
        if llm_extract_schema:
            kwargs["llm_extract"] = {"schema": llm_extract_schema}

        # Call the API
        result = client.scrape.create(**kwargs)
        
        # Extract content
        content = None
        
        # Check for LLM extract result first
        if hasattr(result, "llm_extract") and result.llm_extract:
            import json
            return json.dumps(result.llm_extract, indent=2)
        
        # Check for parser result
        if hasattr(result, "parser_result") and result.parser_result:
            import json
            return json.dumps(result.parser_result, indent=2)
        
        # Return content based on format priority
        if hasattr(result, "markdown") and result.markdown:
            content = result.markdown
        elif hasattr(result, "markdown_content") and result.markdown_content:
            content = result.markdown_content
        elif hasattr(result, "html") and result.html:
            content = result.html
        elif hasattr(result, "html_content") and result.html_content:
            content = result.html_content
        elif hasattr(result, "text") and result.text:
            content = result.text
        elif hasattr(result, "text_content") and result.text_content:
            content = result.text_content
        else:
            content = str(result)
            
        return content

    def _crawl(
        self,
        url: str,
        max_pages: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> str:
        """Crawl a website by following links.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude

        Returns:
            Crawl results summary
        """
        client = self._get_client()
        
        max_pages = max_pages or self.default_max_pages
        max_pages = max(1, min(max_pages, 1000))  # Clamp between 1 and 1000
        
        kwargs: dict[str, Any] = {
            "url": url,
            "max_pages": max_pages,
        }
        
        if include_patterns:
            kwargs["include_patterns"] = include_patterns
        if exclude_patterns:
            kwargs["exclude_patterns"] = exclude_patterns

        # Start the crawl
        result = client.crawl.start(**kwargs)
        
        # Get crawl ID
        crawl_id = getattr(result, "id", None) or getattr(result, "crawl_id", None)
        
        if crawl_id:
            # Try to get pages if available
            pages = []
            if hasattr(result, "pages"):
                pages = result.pages or []
            
            output = [
                "Crawl started successfully!",
                "",
                f"Crawl ID: {crawl_id}",
                f"Starting URL: {url}",
                f"Max pages: {max_pages}",
            ]
            
            if pages:
                output.append(f"Pages crawled: {len(pages)}")
                output.append("")
                output.append("Crawled URLs:")
                for page in pages[:20]:  # Show first 20
                    page_url = getattr(page, "url", str(page))
                    output.append(f"  • {page_url}")
                if len(pages) > 20:
                    output.append(f"  ... and {len(pages) - 20} more")
            
            return "\n".join(output)
        
        return f"Crawl initiated for {url}: {result}"

    def _map(self, url: str) -> str:
        """Map/discover all URLs on a website.

        Args:
            url: Website URL to map

        Returns:
            List of discovered URLs
        """
        client = self._get_client()
        
        # Create sitemap
        result = client.sitemap.create(url=url)
        
        # Extract URLs
        urls = []
        if hasattr(result, "urls") and result.urls:
            urls = result.urls
        elif hasattr(result, "links") and result.links:
            urls = result.links
        elif isinstance(result, dict):
            urls = result.get("urls", []) or result.get("links", [])
        
        if urls:
            output = [
                f"Found {len(urls)} URLs on {url}:",
                "",
            ]
            for u in urls[:100]:  # Show first 100
                output.append(f"  • {u}")
            if len(urls) > 100:
                output.append(f"  ... and {len(urls) - 100} more")
            return "\n".join(output)
        
        return f"No URLs found on {url}"

    def _answer(
        self,
        task: str,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Get AI-powered answer using web search.

        Args:
            task: Question or task to answer
            json_schema: Optional JSON schema for structured output

        Returns:
            AI-generated answer with sources
        """
        import json as json_module
        import os

        import requests
        
        # Get API key
        api_key = self.api_key or os.environ.get("OLOSTEP_API_KEY")
        if not api_key:
            return "Error: OLOSTEP_API_KEY not found"
        
        # Call Answers API directly (not yet in SDK)
        endpoint = "https://api.olostep.com/v1/answers"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload: dict[str, Any] = {"task": task}
        if json_schema:
            payload["json"] = json_schema
        
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Format output
        if "result" in data:
            if isinstance(data["result"], dict):
                return json_module.dumps(data["result"], indent=2)
            return str(data["result"])
        
        return json_module.dumps(data, indent=2)


# Convenience tool classes for specific modes
class OlostepScrapeTool(OlostepTool):
    """Olostep tool pre-configured for scraping."""
    
    name: str = "OlostepScrapeTool"
    description: str = (
        "Scrape content from a webpage. Returns clean markdown content. "
        "Supports LLM extraction with schemas and built-in parsers "
        "(e.g., @olostep/google-search, @olostep/linkedin-profile)."
    )
    args_schema: type[BaseModel] = OlostepScrapeSchema
    default_mode: Literal["scrape", "crawl", "map", "answer"] = "scrape"
    
    def __init__(self, **kwargs):
        kwargs.setdefault("default_mode", "scrape")
        super().__init__(**kwargs)


class OlostepCrawlTool(OlostepTool):
    """Olostep tool pre-configured for crawling."""
    
    name: str = "OlostepCrawlTool"
    description: str = (
        "Crawl a website by following links. Extracts content from multiple pages. "
        "Configure max_pages and URL patterns to control the crawl."
    )
    args_schema: type[BaseModel] = OlostepCrawlSchema
    default_mode: Literal["scrape", "crawl", "map", "answer"] = "crawl"
    
    def __init__(self, **kwargs):
        kwargs.setdefault("default_mode", "crawl")
        super().__init__(**kwargs)


class OlostepMapTool(OlostepTool):
    """Olostep tool pre-configured for mapping/sitemap."""
    
    name: str = "OlostepMapTool"
    description: str = (
        "Discover all URLs on a website. Returns a comprehensive list of pages."
    )
    args_schema: type[BaseModel] = OlostepMapSchema
    default_mode: Literal["scrape", "crawl", "map", "answer"] = "map"
    
    def __init__(self, **kwargs):
        kwargs.setdefault("default_mode", "map")
        super().__init__(**kwargs)


class OlostepAnswerTool(OlostepTool):
    """Olostep tool pre-configured for AI-powered web search."""
    
    name: str = "OlostepAnswerTool"
    description: str = (
        "AI-powered web search that answers questions with real-world data. "
        "Returns structured JSON answers with sources. Use json_schema to "
        "specify the output format."
    )
    args_schema: type[BaseModel] = OlostepAnswerSchema
    default_mode: Literal["scrape", "crawl", "map", "answer"] = "answer"
    
    def __init__(self, **kwargs):
        kwargs.setdefault("default_mode", "answer")
        super().__init__(**kwargs)

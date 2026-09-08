from __future__ import annotations

import logging
import re
from typing import Any, cast

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup, Tag
    import requests

    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False


class WikipediaException(Exception):  # noqa: N818
    pass


class PageError(Exception):
    pass


class DisambiguationError(Exception):
    def __init__(self, title: str, options: list[str]) -> None:
        self.title = title
        self.options = options
        super().__init__(title, options)


class WikipediaPage:
    """Represents a Wikipedia page with title, url, content, and summary attributes."""

    def __init__(self, title: str, url: str, client: WikipediaClient) -> None:
        """Initializes a WikipediaPage instance.

        Args:
            title (str): The title of the Wikipedia page.
            url (str): The full URL of the Wikipedia page.
            client (WikipediaClient): The WikipediaClient instance used for fetching data.
        """
        self.title = title
        self.url = url
        self._client = client
        self._content: str | None = None
        self._summary: str | None = None

    @property
    def content(self) -> str:
        """Retrieves the full plain text content of the page.

        Returns:
            str: The full article content.
        """
        if self._content is None:
            self._content = self._client.get_content(self.title)
        return self._content

    @property
    def summary(self) -> str:
        """Retrieves the lead section summary of the page.

        Returns:
            str: The article summary text.
        """
        if self._summary is None:
            self._summary = self._client.summary(self.title)
        return self._summary


class WikipediaClient:
    """Instance-based Wikipedia API client that avoids mutating global module state."""

    def __init__(
        self,
        lang: str = "en",
        user_agent: str = "CrewAIWikipediaSearchTool/1.0 (https://crewai.com; contact@crewai.com)",
    ) -> None:
        """Initializes a WikipediaClient instance with specified language and user agent.

        Args:
            lang (str): Wikipedia language code (e.g., 'en', 'tr', 'fr'). Defaults to 'en'.
            user_agent (str): User-Agent string sent in HTTP requests.

        Raises:
            ValueError: If the language code contains invalid characters.
        """
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError(
                "The 'beautifulsoup4' and 'requests' packages are required to use WikipediaClient. "
                "Please install them using your package manager (e.g., `pip install beautifulsoup4 requests` or `uv add beautifulsoup4 requests`)."
            )

        if not re.match(r"^[a-z\-]+$", lang.lower()):
            raise ValueError(f"Invalid language code: {lang}")

        self.lang = lang
        self.user_agent = user_agent
        self.api_url = f"https://{self.lang.lower()}.wikipedia.org/w/api.php"

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sends an HTTP GET request to the Wikipedia API.

        Args:
            params (dict[str, Any]): Query parameters for the Wikipedia API.

        Returns:
            dict[str, Any]: Parsed JSON response dictionary.

        Raises:
            WikipediaException: If the API response contains an error field.
        """
        params["format"] = "json"
        if "action" not in params:
            params["action"] = "query"

        headers = {"User-Agent": self.user_agent}
        response = requests.get(
            self.api_url, params=params, headers=headers, timeout=10
        )
        response.raise_for_status()
        data = cast(dict[str, Any], response.json())

        if "error" in data:
            error_info = str(data["error"].get("info", "Unknown Wikipedia API error"))
            raise WikipediaException(error_info)

        return data

    def search(self, query: str, results: int = 3) -> list[str]:
        """Searches Wikipedia for matching article titles.

        Args:
            query (str): The search query string.
            results (int): Maximum number of search result titles to return. Defaults to 3.

        Returns:
            list[str]: A list of matching Wikipedia page titles.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": results,
        }
        data = self._request(params)
        search_items = cast(
            list[dict[str, Any]], data.get("query", {}).get("search", [])
        )
        return [str(item["title"]) for item in search_items]

    def page(self, title: str, auto_suggest: bool = False) -> WikipediaPage:
        """Retrieves page information and constructs a WikipediaPage instance.

        Args:
            title (str): Title of the Wikipedia page.
            auto_suggest (bool): Flag for title suggestion. Defaults to False.

        Returns:
            WikipediaPage: Constructed WikipediaPage object.

        Raises:
            PageError: If the page does not exist.
            DisambiguationError: If the title refers to a disambiguation page.
        """
        params = {
            "action": "query",
            "prop": "info|pageprops",
            "inprop": "url",
            "ppprop": "disambiguation",
            "redirects": "",
            "titles": title,
        }
        data = self._request(params)
        query = cast(dict[str, Any], data.get("query", {}))
        pages = cast(dict[str, Any], query.get("pages", {}))

        if not pages:
            raise PageError(title)

        page_id = next(iter(pages))
        page_info = cast(dict[str, Any], pages[page_id])

        if query.get("redirects"):
            redirect_to = str(query["redirects"][0]["to"])
            return self.page(redirect_to, auto_suggest=False)

        if "missing" in page_info or page_id == "-1":
            raise PageError(title)

        if "pageprops" in page_info and "disambiguation" in page_info["pageprops"]:
            rev_params = {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvparse": "",
                "rvlimit": 1,
                "titles": title,
            }
            rev_data = self._request(rev_params)
            rev_pages = cast(dict[str, Any], rev_data.get("query", {}).get("pages", {}))
            rev_id = next(iter(rev_pages))
            revisions = cast(
                list[dict[str, Any]], rev_pages[rev_id].get("revisions", [])
            )
            html = str(revisions[0].get("*", "")) if revisions else ""

            soup = BeautifulSoup(html, "html.parser")
            may_refer_to = [
                li.a.get_text()
                for li in soup.find_all("li")
                if isinstance(li, Tag) and li.a
            ]
            raise DisambiguationError(title, may_refer_to)

        page_title = str(page_info.get("title", title))
        page_url = str(
            page_info.get(
                "fullurl",
                f"https://{self.lang.lower()}.wikipedia.org/wiki/{page_title}",
            )
        )
        return WikipediaPage(title=page_title, url=page_url, client=self)

    def summary(self, title: str, auto_suggest: bool = False) -> str:
        """Retrieves the lead section summary of a Wikipedia page.

        Args:
            title (str): Title of the Wikipedia page.
            auto_suggest (bool): Flag for title suggestion. Defaults to False.

        Returns:
            str: Plain text lead section summary.
        """
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "",
            "exintro": "",
            "titles": title,
        }
        data = self._request(params)
        pages = cast(dict[str, Any], data.get("query", {}).get("pages", {}))
        if not pages:
            return ""
        page_id = next(iter(pages))
        return cast(str, pages[page_id].get("extract", ""))

    def get_content(self, title: str) -> str:
        """Retrieves the full plain text content of a Wikipedia page.

        Args:
            title (str): Title of the Wikipedia page.

        Returns:
            str: Full plain text content of the page.
        """
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "",
            "titles": title,
        }
        data = self._request(params)
        pages = cast(dict[str, Any], data.get("query", {}).get("pages", {}))
        if not pages:
            return ""
        page_id = next(iter(pages))
        return cast(str, pages[page_id].get("extract", ""))


class WikipediaSearchToolSchema(BaseModel):
    """Input schema for WikipediaSearchTool."""

    search_query: str = Field(
        ...,
        description="Search query to find relevant Wikipedia articles, e.g. 'Artificial Intelligence' or 'Python programming'",
    )
    lang: str | None = Field(
        default=None,
        description="Optional Wikipedia language code (e.g., 'en', 'tr', 'es', 'fr', 'de'). Defaults to 'en'.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional maximum number of Wikipedia search results to return (between 1 and 10). Defaults to 3.",
    )
    load_full_content: bool | None = Field(
        default=None,
        description="Optional flag. If True, returns full article content instead of lead section summary. Defaults to False.",
    )


class WikipediaSearchTool(BaseTool):
    """A tool that searches Wikipedia and retrieves article summaries or full content."""

    name: str = "Search Wikipedia"
    description: str = "A tool to search Wikipedia for information, returning titles, URLs, and summaries or full content of matching articles."
    args_schema: type[BaseModel] = WikipediaSearchToolSchema
    model_config = ConfigDict(extra="allow")

    package_dependencies: list[str] = Field(
        default_factory=lambda: ["beautifulsoup4", "requests"]
    )
    env_vars: list[EnvVar] = Field(default_factory=list)

    lang: str = "en"
    limit: int = Field(default=3, ge=1, le=10)
    load_full_content: bool = False
    user_agent: str = (
        "CrewAIWikipediaSearchTool/1.0 (https://crewai.com; contact@crewai.com)"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initialize_wikipedia()

    def _initialize_wikipedia(self) -> None:
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError(
                "The 'beautifulsoup4' and 'requests' packages are required to use the WikipediaSearchTool. "
                "Please install them using your package manager (e.g., `pip install beautifulsoup4 requests` or `uv add beautifulsoup4 requests`)."
            )

    def _run(
        self,
        search_query: str,
        lang: str | None = None,
        limit: int | None = None,
        load_full_content: bool | None = None,
        **kwargs: Any,
    ) -> str:
        if not WIKIPEDIA_AVAILABLE:
            return (
                "Error: The 'beautifulsoup4' and 'requests' packages are required to use WikipediaSearchTool. "
                "Please install them using your package manager (e.g., `pip install beautifulsoup4 requests` or `uv add beautifulsoup4 requests`)."
            )

        target_lang = lang or self.lang

        # Runtime limit validation & clamping: ensure limit is within [1, 10]
        raw_limit = self.limit if limit is None else limit
        if raw_limit < 1:
            target_limit = 1
        elif raw_limit > 10:
            target_limit = 10
        else:
            target_limit = raw_limit

        should_load_full_content = (
            self.load_full_content if load_full_content is None else load_full_content
        )

        try:
            client: WikipediaClient = kwargs.get("client") or WikipediaClient(
                lang=target_lang, user_agent=self.user_agent
            )
            search_results = client.search(search_query, results=target_limit)
        except Exception as e:
            logger.error("Wikipedia search failed: %s", type(e).__name__)
            return f"Error searching Wikipedia for '{search_query}': {e!s}"

        if not search_results:
            return f"No Wikipedia results found for query: '{search_query}'"

        formatted_results: list[str] = []

        for title in search_results:
            try:
                page = client.page(title, auto_suggest=False)
                if should_load_full_content:
                    body = f"Content: {page.content}"
                else:
                    summary = client.summary(page.title, auto_suggest=False)
                    body = f"Summary: {summary}"

                formatted_results.append(
                    f"Title: {page.title}\nURL: {page.url}\n{body}"
                )
            except DisambiguationError as e:  # noqa: PERF203
                options_str = ", ".join(e.options[:5])
                formatted_results.append(
                    f"Title: {title} (Disambiguation)\n"
                    f"Note: '{title}' refers to multiple topics: {options_str}..."
                )
            except PageError:
                formatted_results.append(
                    f"Title: {title}\nNote: Could not retrieve page details."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch details for Wikipedia page '{title}': {e}"
                )
                formatted_results.append(
                    f"Title: {title}\nNote: Error retrieving details: {e!s}"
                )

        separator = "\n\n" + "-" * 80 + "\n\n"
        return separator.join(formatted_results)


if WIKIPEDIA_AVAILABLE:
    if not getattr(WikipediaSearchTool, "_model_rebuilt", False):
        WikipediaSearchTool.model_rebuild()
        WikipediaSearchTool._model_rebuilt = True  # type: ignore[attr-defined]

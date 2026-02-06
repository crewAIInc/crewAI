from datetime import datetime
import json
import os
import time
from typing import Annotated, Any, ClassVar, Literal

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
import requests


load_dotenv()


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    filename = f"search_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(filename, "w") as file:
        file.write(content)


FreshnessPreset = Literal["pd", "pw", "pm", "py"]
FreshnessRange = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$")
]
Freshness = FreshnessPreset | FreshnessRange
SafeSearch = Literal["off", "moderate", "strict"]


class BraveSearchToolSchema(BaseModel):
    """Input for BraveSearchTool"""

    query: str = Field(..., description="Search query to perform")
    country: str | None = Field(
        default=None,
        description="Country code for geo-targeting (e.g., 'US', 'BR').",
    )
    search_lang: str | None = Field(
        default=None,
        description="Language code for the search results (e.g., 'en', 'es').",
    )
    count: int | None = Field(
        default=None,
        description="The maximum number of results to return. Actual number may be less.",
    )
    offset: int | None = Field(
        default=None, description="Skip the first N result sets/pages. Max is 9."
    )
    safesearch: SafeSearch | None = Field(
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
    operators: bool | None = Field(
        default=None,
        description="Whether to apply search operators (e.g., site:example.com).",
    )


# TODO: Extend support to additional endpoints (e.g., /images, /news, etc.)
class BraveSearchTool(BaseTool):
    """A tool that performs web searches using the Brave Search API."""

    name: str = "Brave Search"
    description: str = (
        "A tool that performs web searches using the Brave Search API. "
        "Results are returned as structured JSON data."
    )
    args_schema: type[BaseModel] = BraveSearchToolSchema
    search_url: str = "https://api.search.brave.com/res/v1/web/search"
    n_results: int = 10
    save_file: bool = False
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BRAVE_API_KEY",
                description="API key for Brave Search",
                required=True,
            ),
        ]
    )
    # Rate limiting parameters
    _last_request_time: ClassVar[float] = 0
    _min_request_interval: ClassVar[float] = 1.0  # seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "BRAVE_API_KEY" not in os.environ:
            raise ValueError(
                "BRAVE_API_KEY environment variable is required for BraveSearchTool"
            )

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        current_time = time.time()
        if (current_time - self._last_request_time) < self._min_request_interval:
            time.sleep(
                self._min_request_interval - (current_time - self._last_request_time)
            )
        BraveSearchTool._last_request_time = time.time()

        # Construct and send the request
        try:
            # Maintain both "search_query" and "query" for backwards compatibility
            query = kwargs.get("search_query") or kwargs.get("query")
            if not query:
                raise ValueError("Query is required")

            payload = {"q": query}

            if country := kwargs.get("country"):
                payload["country"] = country

            if search_lang := kwargs.get("search_lang"):
                payload["search_lang"] = search_lang

            # Fallback to deprecated n_results parameter if no count is provided
            count = kwargs.get("count")
            if count is not None:
                payload["count"] = count
            else:
                payload["count"] = self.n_results

            # Offset may be 0, so avoid truthiness check
            offset = kwargs.get("offset")
            if offset is not None:
                payload["offset"] = offset

            if safesearch := kwargs.get("safesearch"):
                payload["safesearch"] = safesearch

            save_file = kwargs.get("save_file", self.save_file)
            if freshness := kwargs.get("freshness"):
                payload["freshness"] = freshness

            # Boolean parameters
            spellcheck = kwargs.get("spellcheck")
            if spellcheck is not None:
                payload["spellcheck"] = spellcheck

            text_decorations = kwargs.get("text_decorations")
            if text_decorations is not None:
                payload["text_decorations"] = text_decorations

            extra_snippets = kwargs.get("extra_snippets")
            if extra_snippets is not None:
                payload["extra_snippets"] = extra_snippets

            operators = kwargs.get("operators")
            if operators is not None:
                payload["operators"] = operators

            # Limit the result types to "web" since there is presently no
            # handling of other types like "discussions", "faq", "infobox",
            # "news", "videos", or "locations".
            payload["result_filter"] = "web"

            # Setup Request Headers
            headers = {
                "X-Subscription-Token": os.environ["BRAVE_API_KEY"],
                "Accept": "application/json",
            }

            response = requests.get(
                self.search_url, headers=headers, params=payload, timeout=30
            )
            response.raise_for_status()  # Handle non-200 responses
            results = response.json()

            # TODO: Handle other result types like "discussions", "faq", etc.
            web_results_items = []
            if "web" in results:
                web_results = results["web"]["results"]

                for result in web_results:
                    url = result.get("url")
                    title = result.get("title")
                    # If, for whatever reason, this entry does not have a title
                    # or url, skip it.
                    if not url or not title:
                        continue
                    item = {
                        "url": url,
                        "title": title,
                    }
                    description = result.get("description")
                    if description:
                        item["description"] = description
                    snippets = result.get("extra_snippets")
                    if snippets:
                        item["snippets"] = snippets

                    web_results_items.append(item)

            content = json.dumps(web_results_items)
        except requests.RequestException as e:
            return f"Error performing search: {e!s}"
        except KeyError as e:
            return f"Error parsing search results: {e!s}"
        if save_file:
            _save_results_to_file(content)
            return f"\nSearch results: {content}\n"
        return content

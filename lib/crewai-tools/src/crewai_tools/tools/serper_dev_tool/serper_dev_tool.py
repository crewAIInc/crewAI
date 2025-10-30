import datetime
import json
import logging
import os
from typing import Any, TypedDict

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__name__)


class KnowledgeGraph(TypedDict, total=False):
    """Knowledge graph data from search results."""

    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: dict[str, Any]


class Sitelink(TypedDict):
    """Sitelink data for organic search results."""

    title: str
    link: str


class OrganicResult(TypedDict, total=False):
    """Organic search result data."""

    title: str
    link: str
    snippet: str
    position: int | None
    sitelinks: list[Sitelink]


class PeopleAlsoAskResult(TypedDict):
    """People Also Ask result data."""

    question: str
    snippet: str
    title: str
    link: str


class RelatedSearchResult(TypedDict):
    """Related search result data."""

    query: str


class NewsResult(TypedDict):
    """News search result data."""

    title: str
    link: str
    snippet: str
    date: str
    source: str
    imageUrl: str


class SearchParameters(TypedDict, total=False):
    """Search parameters used for the query."""

    q: str
    type: str


class FormattedResults(TypedDict, total=False):
    """Formatted search results from Serper API."""

    searchParameters: SearchParameters
    knowledgeGraph: KnowledgeGraph
    organic: list[OrganicResult]
    peopleAlsoAsk: list[PeopleAlsoAskResult]
    relatedSearches: list[RelatedSearchResult]
    news: list[NewsResult]
    credits: int


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    try:
        filename = f"search_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        with open(filename, "w") as file:
            file.write(content)
        logger.info(f"Results saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save results to file: {e}")
        raise


class SerperDevToolSchema(BaseModel):
    """Input for SerperDevTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class SerperDevTool(BaseTool):
    name: str = "Search the internet with Serper"
    description: str = (
        "A tool that can be used to search the internet with a search_query. "
        "Supports different search types: 'search' (default), 'news'"
    )
    args_schema: type[BaseModel] = SerperDevToolSchema
    base_url: str = "https://google.serper.dev"
    n_results: int = 10
    save_file: bool = False
    search_type: str = "search"
    country: str | None = ""
    location: str | None = ""
    locale: str | None = ""
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPER_API_KEY", description="API key for Serper", required=True
            ),
        ]
    )

    def _get_search_url(self, search_type: str) -> str:
        """Get the appropriate endpoint URL based on search type."""
        search_type = search_type.lower()
        allowed_search_types = ["search", "news"]
        if search_type not in allowed_search_types:
            raise ValueError(
                f"Invalid search type: {search_type}. Must be one of: {', '.join(allowed_search_types)}"
            )
        return f"{self.base_url}/{search_type}"

    @staticmethod
    def _process_knowledge_graph(kg: dict[str, Any]) -> KnowledgeGraph:
        """Process knowledge graph data from search results."""
        return {
            "title": kg.get("title", ""),
            "type": kg.get("type", ""),
            "website": kg.get("website", ""),
            "imageUrl": kg.get("imageUrl", ""),
            "description": kg.get("description", ""),
            "descriptionSource": kg.get("descriptionSource", ""),
            "descriptionLink": kg.get("descriptionLink", ""),
            "attributes": kg.get("attributes", {}),
        }

    def _process_organic_results(
        self, organic_results: list[dict[str, Any]]
    ) -> list[OrganicResult]:
        """Process organic search results."""
        processed_results: list[OrganicResult] = []
        for result in organic_results[: self.n_results]:
            try:
                result_data: OrganicResult = {  # type: ignore[typeddict-item]
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position"),
                }

                if "sitelinks" in result:
                    result_data["sitelinks"] = [  # type: ignore[typeddict-unknown-key]
                        {
                            "title": sitelink.get("title", ""),
                            "link": sitelink.get("link", ""),
                        }
                        for sitelink in result["sitelinks"]
                    ]

                processed_results.append(result_data)
            except KeyError:  # noqa: PERF203
                logger.warning(f"Skipping malformed organic result: {result}")
                continue
        return processed_results  # type: ignore[return-value]

    def _process_people_also_ask(
        self, paa_results: list[dict[str, Any]]
    ) -> list[PeopleAlsoAskResult]:
        """Process 'People Also Ask' results."""
        processed_results: list[PeopleAlsoAskResult] = []
        for result in paa_results[: self.n_results]:
            try:
                result_data: PeopleAlsoAskResult = {  # type: ignore[typeddict-item]
                    "question": result["question"],
                    "snippet": result.get("snippet", ""),
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                }
                processed_results.append(result_data)
            except KeyError:  # noqa: PERF203
                logger.warning(f"Skipping malformed PAA result: {result}")
                continue
        return processed_results  # type: ignore[return-value]

    def _process_related_searches(
        self, related_results: list[dict[str, Any]]
    ) -> list[RelatedSearchResult]:
        """Process related search results."""
        processed_results: list[RelatedSearchResult] = []
        for result in related_results[: self.n_results]:
            try:
                processed_results.append({"query": result["query"]})  # type: ignore[typeddict-item]
            except KeyError:  # noqa: PERF203
                logger.warning(f"Skipping malformed related search result: {result}")
                continue
        return processed_results  # type: ignore[return-value]

    def _process_news_results(
        self, news_results: list[dict[str, Any]]
    ) -> list[NewsResult]:
        """Process news search results."""
        processed_results: list[NewsResult] = []
        for result in news_results[: self.n_results]:
            try:
                result_data: NewsResult = {  # type: ignore[typeddict-item]
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", ""),
                    "imageUrl": result.get("imageUrl", ""),
                }
                processed_results.append(result_data)
            except KeyError:  # noqa: PERF203
                logger.warning(f"Skipping malformed news result: {result}")
                continue
        return processed_results  # type: ignore[return-value]

    def _make_api_request(self, search_query: str, search_type: str) -> dict[str, Any]:
        """Make API request to Serper."""
        search_url = self._get_search_url(search_type)
        payload = {"q": search_query, "num": self.n_results}

        if self.country != "":
            payload["gl"] = self.country
        if self.location != "":
            payload["location"] = self.location
        if self.locale != "":
            payload["hl"] = self.locale

        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "content-type": "application/json",
        }

        response = None
        try:
            response = requests.post(
                search_url, headers=headers, json=payload, timeout=10
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                logger.error("Empty response from Serper API")
                raise ValueError("Empty response from Serper API")
            return results
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to Serper API: {e}"
            if response is not None and hasattr(response, "content"):
                error_msg += f"\nResponse content: {response.content.decode('utf-8', errors='replace')}"
            logger.error(error_msg)
            raise
        except json.JSONDecodeError as e:
            if response is not None and hasattr(response, "content"):
                logger.error(f"Error decoding JSON response: {e}")
                logger.error(
                    f"Response content: {response.content.decode('utf-8', errors='replace')}"
                )
            else:
                logger.error(
                    f"Error decoding JSON response: {e} (No response content available)"
                )
            raise

    def _process_search_results(
        self, results: dict[str, Any], search_type: str
    ) -> dict[str, Any]:
        """Process search results based on search type."""
        formatted_results: dict[str, Any] = {}

        if search_type == "search":
            if "knowledgeGraph" in results:
                formatted_results["knowledgeGraph"] = self._process_knowledge_graph(
                    results["knowledgeGraph"]
                )

            if "organic" in results:
                formatted_results["organic"] = self._process_organic_results(
                    results["organic"]
                )

            if "peopleAlsoAsk" in results:
                formatted_results["peopleAlsoAsk"] = self._process_people_also_ask(
                    results["peopleAlsoAsk"]
                )

            if "relatedSearches" in results:
                formatted_results["relatedSearches"] = self._process_related_searches(
                    results["relatedSearches"]
                )

        elif search_type == "news":
            if "news" in results:
                formatted_results["news"] = self._process_news_results(results["news"])

        return formatted_results

    def _run(self, **kwargs: Any) -> FormattedResults:
        """Execute the search operation."""
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        search_type: str = kwargs.get("search_type", self.search_type)
        save_file = kwargs.get("save_file", self.save_file)

        if not search_query:
            raise ValueError("search_query is required")

        results = self._make_api_request(search_query, search_type)

        formatted_results = {
            "searchParameters": {
                "q": search_query,
                "type": search_type,
                **results.get("searchParameters", {}),
            }
        }

        formatted_results.update(self._process_search_results(results, search_type))
        formatted_results["credits"] = results.get("credits", 1)

        if save_file:
            _save_results_to_file(json.dumps(formatted_results, indent=2))

        return formatted_results  # type: ignore[return-value]

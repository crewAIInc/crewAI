import datetime
import json
import logging
import os
from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    args_schema: Type[BaseModel] = SerperDevToolSchema
    base_url: str = "https://google.serper.dev"
    n_results: int = 10
    save_file: bool = False
    search_type: str = "search"

    def _get_search_url(self, search_type: str) -> str:
        """Get the appropriate endpoint URL based on search type."""
        search_type = search_type.lower()
        allowed_search_types = ["search", "news"]
        if search_type not in allowed_search_types:
            raise ValueError(
                f"Invalid search type: {search_type}. Must be one of: {', '.join(allowed_search_types)}"
            )
        return f"{self.base_url}/{search_type}"

    def _process_knowledge_graph(self, kg: dict) -> dict:
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

    def _process_organic_results(self, organic_results: list) -> list:
        """Process organic search results."""
        processed_results = []
        for result in organic_results[: self.n_results]:
            try:
                result_data = {
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position"),
                }

                if "sitelinks" in result:
                    result_data["sitelinks"] = [
                        {
                            "title": sitelink.get("title", ""),
                            "link": sitelink.get("link", ""),
                        }
                        for sitelink in result["sitelinks"]
                    ]

                processed_results.append(result_data)
            except KeyError:
                logger.warning(f"Skipping malformed organic result: {result}")
                continue
        return processed_results

    def _process_people_also_ask(self, paa_results: list) -> list:
        """Process 'People Also Ask' results."""
        processed_results = []
        for result in paa_results[: self.n_results]:
            try:
                result_data = {
                    "question": result["question"],
                    "snippet": result.get("snippet", ""),
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                }
                processed_results.append(result_data)
            except KeyError:
                logger.warning(f"Skipping malformed PAA result: {result}")
                continue
        return processed_results

    def _process_related_searches(self, related_results: list) -> list:
        """Process related search results."""
        processed_results = []
        for result in related_results[: self.n_results]:
            try:
                processed_results.append({"query": result["query"]})
            except KeyError:
                logger.warning(f"Skipping malformed related search result: {result}")
                continue
        return processed_results

    def _process_news_results(self, news_results: list) -> list:
        """Process news search results."""
        processed_results = []
        for result in news_results[: self.n_results]:
            try:
                result_data = {
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", ""),
                    "imageUrl": result.get("imageUrl", ""),
                }
                processed_results.append(result_data)
            except KeyError:
                logger.warning(f"Skipping malformed news result: {result}")
                continue
        return processed_results

    def _make_api_request(self, search_query: str, search_type: str) -> dict:
        """Make API request to Serper."""
        search_url = self._get_search_url(search_type)
        payload = json.dumps({"q": search_query, "num": self.n_results})
        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "content-type": "application/json",
        }

        response = None
        try:
            response = requests.post(
                search_url, headers=headers, json=json.loads(payload), timeout=10
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
                error_msg += f"\nResponse content: {response.content}"
            logger.error(error_msg)
            raise
        except json.JSONDecodeError as e:
            if response is not None and hasattr(response, "content"):
                logger.error(f"Error decoding JSON response: {e}")
                logger.error(f"Response content: {response.content}")
            else:
                logger.error(
                    f"Error decoding JSON response: {e} (No response content available)"
                )
            raise

    def _process_search_results(self, results: dict, search_type: str) -> dict:
        """Process search results based on search type."""
        formatted_results = {}

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

    def _run(self, **kwargs: Any) -> Any:
        """Execute the search operation."""
        search_query = kwargs.get("search_query") or kwargs.get("query")
        search_type = kwargs.get("search_type", self.search_type)
        save_file = kwargs.get("save_file", self.save_file)

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

        return formatted_results

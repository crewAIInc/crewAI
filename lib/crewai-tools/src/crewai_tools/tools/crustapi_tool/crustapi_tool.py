import json
import logging
import os
from typing import Any, List, Optional, Type

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# CrustAPI returns Google data as clean JSON from one endpoint. The response shape mirrors the
# common SERP-API schema (organic, knowledgeGraph, peopleAlsoAsk, relatedSearches, news, ...),
# so results map straight into an agent's context. Get a free key at https://crustapi.com.

ALLOWED_SEARCH_TYPES = [
    "search",
    "web",
    "news",
    "maps",
    "places",
    "shopping",
    "images",
    "videos",
    "scholar",
    "patents",
]


class CrustApiToolSchema(BaseModel):
    """Input for CrustApiTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search Google"
    )


class CrustApiTool(BaseTool):
    name: str = "Search Google with CrustAPI"
    description: str = (
        "A tool that searches Google via CrustAPI and returns clean, structured JSON. "
        "Supports different search types: 'search' (default web results), 'news', 'maps', "
        "'places', 'shopping', 'images', 'videos', 'scholar', 'patents'."
    )
    args_schema: Type[BaseModel] = CrustApiToolSchema
    base_url: str = "https://crustapi.com/v1"
    n_results: int = 10
    save_file: bool = False
    search_type: str = "search"
    country: Optional[str] = ""
    location: Optional[str] = ""
    locale: Optional[str] = ""
    env_vars: List[EnvVar] = [
        EnvVar(name="CRUSTAPI_API_KEY", description="API key for CrustAPI", required=True),
    ]

    def _normalize_search_type(self, search_type: str) -> str:
        """Validate the search type and map 'search' to CrustAPI's 'web' surface."""
        search_type = (search_type or "search").lower()
        if search_type not in ALLOWED_SEARCH_TYPES:
            raise ValueError(
                f"Invalid search type: {search_type}. "
                f"Must be one of: {', '.join(ALLOWED_SEARCH_TYPES)}"
            )
        return "web" if search_type == "search" else search_type

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
                    "position": result.get("position", ""),
                }
                processed_results.append(result_data)
            except KeyError:
                logger.warning(f"Skipping malformed organic result: {result}")
                continue
        return processed_results

    def _process_people_also_ask(self, paa_results: list) -> list:
        """Process 'people also ask' results."""
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
        """Make an API request to CrustAPI."""
        params = {"type": search_type, "q": search_query, "num": self.n_results}

        if self.country != "":
            params["gl"] = self.country
        if self.location != "":
            params["location"] = self.location
        if self.locale != "":
            params["hl"] = self.locale

        headers = {"x-api-key": os.environ["CRUSTAPI_API_KEY"]}

        response = None
        try:
            response = requests.get(
                f"{self.base_url}/search", headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            results = response.json()
            if not results:
                logger.error("Empty response from CrustAPI")
                raise ValueError("Empty response from CrustAPI")
            return results
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to CrustAPI: {e}"
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
        """Format known result blocks; pass others through untouched."""
        formatted_results = {}

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
        if "news" in results:
            formatted_results["news"] = self._process_news_results(results["news"])

        # Surfaces like maps, places, shopping, images, videos already come back as clean,
        # named arrays; forward whichever ones are present so the agent gets everything.
        for key in ("places", "shopping", "images", "videos", "reviews"):
            if key in results:
                formatted_results[key] = results[key][: self.n_results]

        return formatted_results

    def _run(self, **kwargs: Any) -> Any:
        """Execute the search operation."""
        search_query = kwargs.get("search_query") or kwargs.get("query")
        search_type = self._normalize_search_type(
            kwargs.get("search_type", self.search_type)
        )
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
            import datetime

            filename = (
                f"crustapi_results_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            try:
                with open(filename, "w") as file:
                    file.write(json.dumps(formatted_results, indent=2))
                logger.info(f"Results saved to {filename}")
            except IOError as e:
                logger.error(f"Failed to save results to file: {e}")

        return formatted_results

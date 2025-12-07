import datetime
import json
import logging
import os
from typing import Any, Literal, TypedDict

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FormattedResults(TypedDict, total=False):
    """Formatted search results from Cloro API."""

    # Google / Search
    organic: list[dict[str, Any]]
    peopleAlsoAsk: list[dict[str, Any]]
    relatedSearches: list[dict[str, Any]]
    ai_overview: dict[str, Any]
    
    # LLM / Common
    text: str
    sources: list[dict[str, Any]]
    
    # Rich Content (Perplexity / ChatGPT)
    shopping_cards: list[dict[str, Any]]
    hotels: list[dict[str, Any]]
    places: list[dict[str, Any]]
    videos: list[dict[str, Any]]
    images: list[dict[str, Any]]
    related_queries: list[str]
    entities: list[dict[str, Any]]
    
    credits: int


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    try:
        filename = f"cloro_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, "w") as file:
            file.write(content)
        logger.info(f"Results saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save results to file: {e}")
        raise


class CloroDevToolSchema(BaseModel):
    """Input for CloroDevTool."""

    search_query: str = Field(
        ..., description="Mandatory query/prompt you want to use to search/query the model"
    )


class CloroTool(BaseTool):
    name: str = "Search/Query with Cloro"
    description: str = (
        "A tool that can be used to search the internet or query LLMs using cloro API. "
        "Supports engines: google, chatgpt, gemini, copilot, perplexity, aimode."
    )
    args_schema: type[BaseModel] = CloroDevToolSchema
    base_url: str = "https://api.cloro.dev/v1/monitor"
    engine: Literal[
        "google",
        "chatgpt",
        "gemini",
        "copilot",
        "perplexity",
        "aimode",
    ] = "google"
    country: str = "US"
    device: str = "desktop"
    pages: int = 1
    save_file: bool = False
    api_key: str | None = Field(None, description="cloro API key")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CLORO_API_KEY", description="API key for cloro", required=True
            ),
        ]
    )

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if api_key:
            self.api_key = api_key

    def _get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("CLORO_API_KEY")
        if env_key:
            return env_key
        raise ValueError("cloro API key not found. Set CLORO_API_KEY environment variable or pass 'api_key' to constructor.")

    def _get_endpoint(self) -> str:
        return f"{self.base_url}/{self.engine}"

    def _make_api_request(self, query: str) -> dict[str, Any]:
        endpoint = self._get_endpoint()
        
        payload: dict[str, Any] = {
            "country": self.country,
        }

        if self.engine == "google":
            payload["query"] = query
            payload["device"] = self.device
            payload["pages"] = self.pages
            payload["include"] = {
                "html": False,
                "aioverview": {"markdown": True}
            }
        else:
            payload["prompt"] = query

        if self.engine in ["chatgpt", "gemini", "copilot", "perplexity", "aimode"]:
            payload["include"] = {"markdown": True}

        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

        response = None
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to cloro API ({self.engine}): {e}"
            if response is not None and hasattr(response, "content"):
                error_msg += f"\nResponse content: {response.content.decode('utf-8', errors='replace')}"
            logger.error(error_msg)
            raise

    def _run(self, **kwargs: Any) -> FormattedResults:
        """Execute the search/query operation."""
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        save_file = kwargs.get("save_file", self.save_file)

        if not search_query:
            raise ValueError("search_query is required")

        api_response = self._make_api_request(search_query)

        if not api_response.get("success"):
            raise ValueError(f"cloro API returned unsuccessful response: {api_response}")
        
        result = api_response.get("result", {})
        formatted_results: FormattedResults = {} # type: ignore

        # Process Google Search Results
        if self.engine == "google":
            if "organicResults" in result:
                formatted_results["organic"] = result["organicResults"]
            if "peopleAlsoAsk" in result:
                formatted_results["peopleAlsoAsk"] = result["peopleAlsoAsk"]
            if "relatedSearches" in result:
                formatted_results["relatedSearches"] = result["relatedSearches"]
            if "aioverview" in result:
                formatted_results["ai_overview"] = result["aioverview"]

        # Process LLM Results
        else:
            if "text" in result:
                formatted_results["text"] = result["text"]
            if "sources" in result:
                formatted_results["sources"] = result["sources"]
            
            # Map rich content if available
            if "shopping_cards" in result:
                formatted_results["shopping_cards"] = result["shopping_cards"]
            elif "shoppingCards" in result:
                formatted_results["shopping_cards"] = result["shoppingCards"]
            
            if "hotels" in result:
                formatted_results["hotels"] = result["hotels"]
            if "places" in result:
                formatted_results["places"] = result["places"]
            if "videos" in result:
                formatted_results["videos"] = result["videos"]
            if "images" in result:
                formatted_results["images"] = result["images"]
            
            if "related_queries" in result:
                formatted_results["related_queries"] = result["related_queries"]
            elif "relatedQueries" in result:
                formatted_results["related_queries"] = result["relatedQueries"]
                
            if "entities" in result:
                formatted_results["entities"] = result["entities"]

        if save_file:
            _save_results_to_file(json.dumps(formatted_results, indent=2))

        return formatted_results
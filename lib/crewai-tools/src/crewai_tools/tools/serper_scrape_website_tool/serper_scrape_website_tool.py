import json
import os

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SerperScrapeWebsiteInput(BaseModel):
    """Input schema for SerperScrapeWebsite."""

    url: str = Field(..., description="The URL of the website to scrape")
    include_markdown: bool = Field(
        default=True,
        description="Whether to include markdown formatting in the scraped content",
    )


class SerperScrapeWebsiteTool(BaseTool):
    name: str = "serper_scrape_website"
    description: str = (
        "Scrapes website content using Serper's scraping API. "
        "This tool can extract clean, readable content from any website URL, "
        "optionally including markdown formatting for better structure."
    )
    args_schema: type[BaseModel] = SerperScrapeWebsiteInput
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPER_API_KEY", description="API key for Serper", required=True
            ),
        ]
    )

    def _run(self, url: str, include_markdown: bool = True) -> str:
        """Scrape website content using Serper API.

        Args:
            url: The URL to scrape
            include_markdown: Whether to include markdown formatting

        Returns:
            Scraped website content as a string
        """
        try:
            # Serper API endpoint
            api_url = "https://scrape.serper.dev"

            # Get API key from environment variable for security
            api_key = os.getenv("SERPER_API_KEY")

            # Prepare the payload
            payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

            # Set headers
            headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

            # Make the API request
            response = requests.post(
                api_url,
                headers=headers,
                data=payload,
                timeout=30,
            )

            # Check if request was successful
            if response.status_code == 200:
                result = response.json()

                # Extract the scraped content
                if "text" in result:
                    return result["text"]
                return f"Successfully scraped {url}, but no text content found in response: {response.text}"
            return (
                f"Error scraping {url}: HTTP {response.status_code} - {response.text}"
            )

        except requests.exceptions.RequestException as e:
            return f"Network error while scraping {url}: {e!s}"
        except json.JSONDecodeError as e:
            return f"Error parsing JSON response while scraping {url}: {e!s}"
        except Exception as e:
            return f"Unexpected error while scraping {url}: {e!s}"

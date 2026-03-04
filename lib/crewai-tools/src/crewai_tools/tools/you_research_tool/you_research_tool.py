import json
import os
from typing import Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class YouResearchToolSchema(BaseModel):
    """Input for YouResearchTool."""

    input: str = Field(
        ...,
        description="The research question or complex query requiring in-depth investigation and multi-step reasoning.",
        max_length=40000,
    )
    research_effort: Literal["lite", "standard", "deep", "exhaustive"] = Field(
        default="standard",
        description=(
            "Controls how much time and effort the Research API spends on your question. "
            "lite: fast answers, standard: balanced (default), deep: thorough, exhaustive: most comprehensive."
        ),
    )


class YouResearchTool(BaseTool):
    """A tool that performs comprehensive research using the You.com Research API."""

    name: str = "You.com Research"
    description: str = (
        "Performs comprehensive, multi-step research using the You.com Research API. "
        "Unlike a simple web search, it runs multiple searches, reads through sources, "
        "and synthesizes everything into a thorough, well-cited answer. "
        "Best for complex questions that require in-depth investigation."
    )
    args_schema: type[BaseModel] = YouResearchToolSchema
    research_url: str = "https://api.you.com/v1/research"
    research_effort: Literal["lite", "standard", "deep", "exhaustive"] = "standard"
    timeout: int = 120
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="YOU_API_KEY",
                description="API key for You.com research service",
                required=True,
            ),
        ],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "YOU_API_KEY" not in os.environ:
            raise ValueError(
                "YOU_API_KEY environment variable is required for YouResearchTool. "
                "Get your API key at https://you.com/platform/api-keys",
            )

    def _run(
        self,
        input: str,
        research_effort: Literal["lite", "standard", "deep", "exhaustive"] = "standard",
    ) -> str:
        """Execute the research operation.

        Args:
            input: The research question or complex query.
            research_effort: Effort level controlling depth vs. speed.

        Returns:
            JSON string containing the research answer and sources.
        """
        try:
            if not input:
                raise ValueError("Input is required")

            payload = {
                "input": input,
                "research_effort": research_effort,
            }

            headers = {
                "X-API-Key": os.environ["YOU_API_KEY"],
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.research_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()

            return json.dumps(results, indent=2)

        except requests.RequestException as e:
            return f"Error performing research: {e!s}"
        except ValueError as e:
            return f"Invalid parameters: {e!s}"
        except KeyError as e:
            return f"Error parsing research results: {e!s}"

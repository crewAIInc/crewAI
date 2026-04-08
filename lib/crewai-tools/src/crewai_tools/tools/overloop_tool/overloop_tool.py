import json
import subprocess
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class OverloopProspectSearchSchema(BaseModel):
    """Input for OverloopProspectSearchTool."""

    query: str = Field(
        ...,
        description=(
            "Search criteria for prospects as a JSON string. "
            "Supports filters: job_titles, locations, company_sizes, industries, keywords. "
            'Example: \'{"job_titles": ["CTO", "VP Engineering"], "locations": ["United States"]}\''
        ),
    )


class OverloopProspectSearchTool(BaseTool):
    """
    OverloopProspectSearchTool - Search and source B2B prospects using Overloop CLI.

    Searches a 450M+ contact database by job title, location, company size,
    industry, and keywords. Returns estimated counts and prospect details.

    Dependencies:
        - overloop-cli (npm install -g overloop-cli)
        - An Overloop account (overloop login)
    """

    name: str = "Overloop Prospect Search"
    description: str = (
        "Search and source B2B prospects using Overloop CLI. "
        "Find contacts by job title, location, company size, and industry "
        "from a 450M+ contact database."
    )
    args_schema: Type[BaseModel] = OverloopProspectSearchSchema

    def _run(self, **kwargs: Any) -> Any:
        query = kwargs.get("query", "")
        try:
            result = subprocess.run(
                ["overloop", "sourcings:estimate", "--search-criteria", query],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"Error running Overloop CLI: {result.stderr.strip()}"
            return result.stdout.strip() or "No results found."
        except FileNotFoundError:
            return (
                "Overloop CLI not found. Install it with: npm install -g overloop-cli"
            )
        except subprocess.TimeoutExpired:
            return "Overloop CLI timed out after 30 seconds."


class OverloopCampaignSchema(BaseModel):
    """Input for OverloopCampaignTool."""

    action: str = Field(
        ...,
        description=(
            "Overloop CLI command to execute. "
            "Examples: 'campaigns:list', 'campaigns:show --id 123', "
            "'campaigns:create --name \"Q1 Outbound\" --steps email,linkedin'"
        ),
    )


class OverloopCampaignTool(BaseTool):
    """
    OverloopCampaignTool - Create and manage outbound campaigns using Overloop CLI.

    Supports multi-channel campaigns across email and LinkedIn, including
    campaign creation, step management, contact enrollment, and status tracking.

    Dependencies:
        - overloop-cli (npm install -g overloop-cli)
        - An Overloop account (overloop login)
    """

    name: str = "Overloop Campaign Manager"
    description: str = (
        "Create and manage multi-channel outbound campaigns (email + LinkedIn) "
        "using Overloop CLI. Supports campaign creation, enrollment, and tracking."
    )
    args_schema: Type[BaseModel] = OverloopCampaignSchema

    def _run(self, **kwargs: Any) -> Any:
        action = kwargs.get("action", "")
        parts = action.split()
        if not parts:
            return "No action specified. Example: 'campaigns:list'"
        try:
            result = subprocess.run(
                ["overloop"] + parts,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"Error running Overloop CLI: {result.stderr.strip()}"
            return result.stdout.strip() or "Command completed with no output."
        except FileNotFoundError:
            return (
                "Overloop CLI not found. Install it with: npm install -g overloop-cli"
            )
        except subprocess.TimeoutExpired:
            return "Overloop CLI timed out after 30 seconds."


class SignalsIntentMonitorSchema(BaseModel):
    """Input for SignalsIntentMonitorTool."""

    business_id: str = Field(
        default="1",
        description="Signals business ID to fetch leads from.",
    )
    per_page: int = Field(
        default=20,
        description="Number of leads to return per page.",
    )


class SignalsIntentMonitorTool(BaseTool):
    """
    SignalsIntentMonitorTool - Monitor buying intent signals using Signals CLI.

    Tracks LinkedIn engagers, keyword posters, job changers, and funding events
    to surface high-intent prospects before they enter an active buying cycle.

    Dependencies:
        - signals-sortlist-cli (npm install -g signals-sortlist-cli)
        - A Signals account (signals login)
    """

    name: str = "Signals Intent Monitor"
    description: str = (
        "Monitor buying intent signals: LinkedIn engagers, keyword posters, "
        "job changers, and funding events. Surfaces high-intent prospects "
        "from Signals CLI."
    )
    args_schema: Type[BaseModel] = SignalsIntentMonitorSchema

    def _run(self, **kwargs: Any) -> Any:
        business_id = kwargs.get("business_id", "1")
        per_page = str(kwargs.get("per_page", 20))
        try:
            result = subprocess.run(
                [
                    "signals",
                    "leads:list",
                    "--business",
                    business_id,
                    "--per-page",
                    per_page,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"Error running Signals CLI: {result.stderr.strip()}"
            return result.stdout.strip() or "No leads found."
        except FileNotFoundError:
            return (
                "Signals CLI not found. Install it with: "
                "npm install -g signals-sortlist-cli"
            )
        except subprocess.TimeoutExpired:
            return "Signals CLI timed out after 30 seconds."

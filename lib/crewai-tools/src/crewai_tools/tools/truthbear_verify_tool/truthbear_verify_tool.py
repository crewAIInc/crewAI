import json
from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TruthBearVerifyToolInput(BaseModel):
    """Input schema for TruthBearVerifyTool."""

    query: str = Field(
        ...,
        description="The fact or data point to verify, e.g. 'US unemployment rate' or 'California earthquake magnitude'",
    )


class TruthBearVerifyTool(BaseTool):
    """
    TruthBearVerifyTool - Verify official facts with Bitcoin-anchored proof.

    Queries 180+ official data signals (FRED, USGS, SEC EDGAR, NOAA, EPA, etc.)
    and returns verifiable results via the free preview endpoint. Each record
    includes the source URL, a SHA-256 record_hash anchored to Bitcoin via
    OpenTimestamps, and a freshness stamp.

    No API key or signup required.
    """

    name: str = "Truth Bear Verify"
    description: str = (
        "Verify official facts from government and institutional sources "
        "(FRED, USGS, SEC EDGAR, NOAA, EPA, 180+ signals). Returns the "
        "official value, source URL, and a SHA-256 record_hash anchored "
        "to Bitcoin for independent re-verification."
    )
    args_schema: Type[BaseModel] = TruthBearVerifyToolInput
    base_url: str = "https://api.truthbear.co"

    def _run(self, query: str, **_: Any) -> str:
        try:
            response = requests.get(
                f"{self.base_url}/trust/preview",
                params={"q": query},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                return json.dumps(data, indent=2)

            parts = []
            if "value" in data:
                parts.append(f"Value: {data['value']}")
            if "source_url" in data:
                parts.append(f"Source: {data['source_url']}")
            if "record_hash" in data:
                parts.append(f"Record Hash: {data['record_hash']}")
            if "freshness" in data:
                parts.append(f"Freshness: {data['freshness']}")
            if "signal" in data:
                parts.append(f"Signal: {data['signal']}")

            if parts:
                return "\n".join(parts)
            return json.dumps(data, indent=2)

        except requests.RequestException as e:
            return f"Error verifying fact: {str(e)}"

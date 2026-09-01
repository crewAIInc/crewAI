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
    TruthBearVerifyTool - Free preview of official-data fact checking.

    Queries the free `/trust/preview` endpoint to retrieve a single-source
    summary from 180+ official data signals (FRED, USGS, SEC EDGAR, NOAA,
    EPA, etc.).  The response includes the matched signal, its value,
    the primary source URL, a SHA-256 record_hash, and a freshness stamp.

    This is a screening-level preview — not decision-grade verification.
    No API key or signup required.
    """

    name: str = "Truth Bear Verify"
    description: str = (
        "Free preview: look up an official data point from government and "
        "institutional sources (FRED, USGS, SEC EDGAR, NOAA, EPA, 180+ "
        "signals). Returns the value, source URL, SHA-256 record_hash, "
        "and freshness stamp. Screening-level only."
    )
    args_schema: Type[BaseModel] = TruthBearVerifyToolInput
    base_url: str = "https://api.truthbear.co"

    def _run(self, query: str, **_: Any) -> str:
        """Query the free /trust/preview endpoint and return the matched fact."""
        try:
            response = requests.get(
                f"{self.base_url}/trust/preview",
                params={"q": query},
                timeout=30,
            )
            response.raise_for_status()
        except requests.Timeout:
            return "Error verifying fact: request timed out after 30 s"
        except requests.ConnectionError:
            return "Error verifying fact: unable to connect to api.truthbear.co"
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            return f"Error verifying fact: HTTP {status}"
        except requests.RequestException as e:
            return f"Error verifying fact: {e}"

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError):
            return "Error verifying fact: response is not valid JSON"

        if not isinstance(data, dict):
            return json.dumps(data, indent=2)

        required = {"source_url": str, "record_hash": str, "freshness": str}
        missing = [k for k, t in required.items() if k not in data or not isinstance(data[k], t)]
        if missing:
            return json.dumps(data, indent=2)

        value = data.get("value")
        if value is None or (isinstance(value, str) and not value.strip()):
            return json.dumps(data, indent=2)

        parts = []
        if "signal" in data:
            parts.append(f"Signal: {data['signal']}")
        parts.append(f"Value: {value}")
        parts.append(f"Source: {data['source_url']}")
        parts.append(f"Record Hash: {data['record_hash']}")
        parts.append(f"Freshness: {data['freshness']}")

        return "\n".join(parts)

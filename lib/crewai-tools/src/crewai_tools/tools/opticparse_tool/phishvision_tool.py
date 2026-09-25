import json
import logging
import os
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PhishVisionToolSchema(BaseModel):
    """Input schema for PhishVisionTool."""

    website_url: str = Field(
        ...,
        description="Mandatory website URL or bare domain (e.g. 'example.com' or 'https://example.com') to audit for cybersecurity threats and phishing."
    )


class PhishVisionTool(BaseTool):
    """Tool for real-time cybersecurity inspection and zero-day phishing detection.

    Audits web pages and domains for credential harvesting, wallet drainers, and brand impersonation.
    """

    name: str = "PhishVision Threat Scanner"
    description: str = (
        "Inspect and audit any web page or domain for real-time security threats, zero-day phishing, "
        "brand impersonation, and smart contract wallet drainers. Returns threat verdict and security score."
    )
    args_schema: Type[BaseModel] = PhishVisionToolSchema
    api_key: Optional[str] = None
    portal_url: str = "https://opticparse-mcp-portal.parastejpal987.workers.dev"
    timeout: int = 15
    env_vars: List[EnvVar] = [
        EnvVar(name="OPTICPARSE_API_KEY", description="API key for OpticParse & PhishVision platform", required=False),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        portal_url: Optional[str] = None,
        timeout: int = 15,
        **kwargs: Any,
    ):
        """Initialize PhishVisionTool."""
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPTICPARSE_API_KEY", "")
        if portal_url:
            cleaned_url = portal_url.rstrip("/")
            if self.api_key and cleaned_url.startswith("http://"):
                raise ValueError("Insecure HTTP portal_url is prohibited when an API key is configured. Please use HTTPS.")
            self.portal_url = cleaned_url
        self.timeout = timeout

    def _normalize_and_validate_url(self, raw_url: str) -> Optional[str]:
        """Normalize bare domains to https:// and validate URL structure."""
        url = raw_url.strip()
        if not url:
            return None
        # Support bare domains like 'example.com' or 'sub.example.com/path'
        if not url.startswith("http://") and not url.startswith("https://"):
            if "://" not in url:
                url = f"https://{url}"
            else:
                return None
        try:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                return url
            return None
        except Exception:
            return None

    def _run(self, website_url: str, **kwargs: Any) -> str:
        """Execute threat intelligence security inspection."""
        normalized_url = self._normalize_and_validate_url(website_url)
        if not normalized_url:
            return f"Error: Invalid or malformed URL: {website_url.strip()}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-PhishVision-Tool/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        payload = {"url": normalized_url}

        try:
            endpoint = f"{self.portal_url}/phishvision/scan"
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
            )

            # Reject redirects to protect credentials
            if 300 <= response.status_code < 400:
                return f"Error: Unexpected gateway redirect (HTTP {response.status_code}). Redirects are disabled for security."

            # Fallback to MCP JSON-RPC call if REST route returns 404/405
            if response.status_code in (404, 405):
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "phishvision_detect",
                        "arguments": {"url": normalized_url},
                    },
                }
                response = requests.post(
                    f"{self.portal_url}/mcp",
                    json=rpc_payload,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                if 300 <= response.status_code < 400:
                    return f"Error: Unexpected gateway redirect (HTTP {response.status_code}). Redirects are disabled for security."

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], dict) and "content" in data["result"]:
                    content = data["result"]["content"]
                    if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                        return content[0]["text"]
                return json.dumps(data, indent=2)
            return str(data)

        except requests.Timeout:
            return f"Error: Security audit timed out for {normalized_url}."
        except requests.HTTPError as e:
            return f"Error: PhishVision inspection returned HTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            logger.error(f"PhishVisionTool execution error: {e}")
            return f"Error: Failed to perform security audit: {str(e)}"

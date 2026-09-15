import json
import logging
import os
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OpticParseToolSchema(BaseModel):
    """Input schema for OpticParseTool."""

    website_url: str = Field(
        ...,
        description="Mandatory target website URL to scrape and extract content from."
    )
    extraction_query: str = Field(
        ...,
        description="Natural language instructions detailing what data fields, tables, or facts to extract."
    )
    response_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema definition to enforce a strict structured output format."
    )


class OpticParseTool(BaseTool):
    """Tool for AI Multimodal Vision web scraping and structured data extraction.

    Extracts clean, token-optimized data from dynamic web applications, single-page apps (SPAs),
    and pages protected by anti-bot verification without relying on fragile CSS/XPath selectors.
    """

    name: str = "OpticParse Web Scraper"
    description: str = (
        "Extract structured, token-optimized data from any live web page using AI Multimodal Vision. "
        "Bypasses bot protections, Turnstile challenges, and dynamic JavaScript rendering. "
        "Takes 'website_url' and 'extraction_query' as required arguments."
    )
    args_schema: Type[BaseModel] = OpticParseToolSchema
    api_key: Optional[str] = None
    portal_url: str = "https://opticparse-mcp-portal.parastejpal987.workers.dev"
    timeout: int = 35
    env_vars: List[EnvVar] = [
        EnvVar(name="OPTICPARSE_API_KEY", description="API key for OpticParse platform access", required=False),
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        portal_url: Optional[str] = None,
        timeout: int = 35,
        **kwargs: Any,
    ):
        """Initialize OpticParseTool.

        Args:
            api_key (Optional[str]): OpticParse API key for high-concurrency requests.
            portal_url (Optional[str]): Custom portal or self-hosted gateway URL.
            timeout (int): Request timeout in seconds. Defaults to 35.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPTICPARSE_API_KEY", "")
        if portal_url:
            self.portal_url = portal_url.rstrip("/")
        self.timeout = timeout

    def _validate_url(self, url: str) -> bool:
        """Validate URL structure."""
        try:
            parsed = urlparse(url.strip())
            return bool(parsed.scheme in ("http", "https") and parsed.netloc)
        except Exception:
            return False

    def _run(
        self,
        website_url: str,
        extraction_query: str,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Execute web scraping and structured extraction.

        Args:
            website_url: Target URL to scrape.
            extraction_query: Natural language instructions for extraction.
            response_schema: Optional JSON schema for typed output.

        Returns:
            Extracted content as formatted JSON or Markdown string.
        """
        url = website_url.strip()
        if not self._validate_url(url):
            return f"Error: Invalid or malformed URL: {url}"

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-OpticParse-Tool/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        payload: Dict[str, Any] = {
            "url": url,
            "query": extraction_query,
        }
        if response_schema:
            payload["response_schema"] = response_schema

        try:
            endpoint = f"{self.portal_url}/tools/scrape"
            response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)

            # Fallback to MCP JSON-RPC standard call if REST route redirects
            if response.status_code in (404, 405):
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "opticparse_scrape",
                        "arguments": {
                            "target_url": url,
                            "extraction_query": extraction_query,
                            "response_schema": response_schema,
                        },
                    },
                }
                response = requests.post(f"{self.portal_url}/mcp", json=rpc_payload, headers=headers, timeout=self.timeout)

            response.raise_for_status()
            data = response.json()

            # Normalize output
            if isinstance(data, dict):
                if "result" in data and isinstance(data["result"], dict) and "content" in data["result"]:
                    content = data["result"]["content"]
                    if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                        return content[0]["text"]
                return json.dumps(data, indent=2)
            return str(data)

        except requests.Timeout:
            return f"Error: Request timed out while extracting content from {url}."
        except requests.HTTPError as e:
            return f"Error: OpticParse gateway returned HTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            logger.error(f"OpticParseTool execution error: {e}")
            return f"Error: Failed to extract data: {str(e)}"

import os
from typing import Any, Optional, Type
from urllib.parse import urlparse
import requests

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class OpticParseToolSchema(BaseModel):
    """Input schema for the OpticParse web scraping tool."""

    url: str = Field(
        description="The target website URL or domain to scrape using multimodal vision."
    )
    query: Optional[str] = Field(
        default="",
        description="Optional natural language extraction query for structured parsing."
    )


class OpticParseTool(BaseTool):
    """Tool for multimodal vision-based web scraping and structured extraction.

    Extracts clean, token-optimized Markdown and structured JSON from dynamic web
    pages. Bypasses anti-bot challenges and renders client-side SPAs without brittle
    CSS or XPath selectors.
    """

    name: str = "OpticParse Multimodal Web Scraper"
    description: str = (
        "Extract structured data and clean Markdown from any website using multimodal vision. "
        "Bypasses Cloudflare Turnstile, anti-bot protections, and renders dynamic JavaScript SPAs without brittle CSS selectors."
    )
    args_schema: Type[BaseModel] = OpticParseToolSchema
    api_key: Optional[str] = os.getenv("OPTICPARSE_API_KEY")
    portal_url: str = os.getenv("OPTICPARSE_PORTAL_URL", "https://opticparse-api.onrender.com")
    timeout: int = 45
    env_vars: list[EnvVar] = [
        EnvVar(
            name="OPTICPARSE_API_KEY",
            description="Optional API key for higher rate limits. Free trial quota available by default.",
            required=False,
        ),
        EnvVar(
            name="OPTICPARSE_PORTAL_URL",
            description="OpticParse gateway portal URL (default: https://opticparse-api.onrender.com)",
            required=False,
        ),
    ]

    def _normalize_url(self, target: str) -> str:
        """Normalize target URL, preserving existing schemes case-insensitively.

        Args:
            target: The input URL or domain string.

        Returns:
            A normalized URL string with https scheme if none was provided.

        Raises:
            ValueError: If the target is empty or lacks a valid network location.
        """
        clean = (target or "").strip()
        if not clean:
            raise ValueError("Target URL or domain cannot be empty.")

        parsed = urlparse(clean)
        if not parsed.scheme:
            clean = f"https://{clean}"
            parsed = urlparse(clean)

        scheme_lower = parsed.scheme.lower()
        if scheme_lower not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid target URL or domain: {target}")
        return clean

    def _run(self, url: str, query: Optional[str] = "") -> str:
        """Execute the web scraping request against the configured OpticParse portal.

        Args:
            url: Target URL to scrape and parse.
            query: Optional extraction query to guide structured data parsing.

        Returns:
            Extracted Markdown or JSON string representation of the target page.

        Raises:
            ValueError: If portal_url does not use HTTPS.
        """
        normalized_url = self._normalize_url(url)
        portal = (self.portal_url or "https://opticparse-api.onrender.com").strip().rstrip("/")
        api_key = (self.api_key or "").strip()

        if not portal.lower().startswith("https://"):
            raise ValueError("portal_url must use a secure HTTPS endpoint to prevent cleartext data transmission.")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-OpticParse-Tool/1.0.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key

        payload = {"url": normalized_url}
        if query:
            payload["query"] = query

        try:
            resp = requests.post(
                f"{portal}/scrape",
                json=payload,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
            )
            if 300 <= resp.status_code < 400:
                return "Error: Gateway redirected. Aborting to protect credentials."
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("markdown") or data.get("extracted_data") or str(data)
            return str(data)
        except Exception as e:
            return f"OpticParse execution error: {str(e)}"

import os
from typing import Any, Optional, Type
import requests
from urllib.parse import urlparse

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class OpticParseToolSchema(BaseModel):
    url: str = Field(description="The target website URL or bare domain to scrape using multimodal vision.")
    query: Optional[str] = Field(
        default="",
        description="Optional natural language extraction query for structured parsing."
    )


class OpticParseTool(BaseTool):
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
        clean = (target or "").strip()
        if not clean:
            raise ValueError("Target URL or domain cannot be empty.")
        if not clean.startswith(("http://", "https://")):
            clean = f"https://{clean}"
        parsed = urlparse(clean)
        if not (parsed.scheme in ("http", "https") and parsed.netloc):
            raise ValueError(f"Invalid target URL or domain: {target}")
        return clean

    def _run(self, url: str, query: Optional[str] = "") -> str:
        normalized_url = self._normalize_url(url)
        portal = (self.portal_url or "https://opticparse-api.onrender.com").strip().rstrip("/")
        api_key = (self.api_key or "").strip()

        if api_key and portal.startswith("http://"):
            raise ValueError("Insecure HTTP portal URL not allowed when API key is set. Use HTTPS.")

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
            if resp.status_code in (301, 302, 307, 308):
                return "Error: Gateway redirected. Aborting to protect credentials."
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("markdown") or data.get("extracted_data") or str(data)
            return str(data)
        except Exception as e:
            return f"OpticParse execution error: {str(e)}"

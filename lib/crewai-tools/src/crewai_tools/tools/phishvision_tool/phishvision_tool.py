import os
from typing import Any, Optional, Type
import requests
from urllib.parse import urlparse

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class PhishVisionToolSchema(BaseModel):
    target: str = Field(description="The target URL or bare domain to audit for cybersecurity threats.")


class PhishVisionTool(BaseTool):
    name: str = "PhishVision Threat & Drainer Shield"
    description: str = (
        "Audit any website or domain in real-time for zero-day phishing, credential harvesting, "
        "brand impersonation, and malicious crypto smart contract drainers before autonomous agent interaction."
    )
    args_schema: Type[BaseModel] = PhishVisionToolSchema
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

    def _run(self, target: str) -> str:
        normalized_url = self._normalize_url(target)
        portal = (self.portal_url or "https://opticparse-api.onrender.com").strip().rstrip("/")
        api_key = (self.api_key or "").strip()

        if api_key and portal.startswith("http://"):
            raise ValueError("Insecure HTTP portal URL not allowed when API key is set. Use HTTPS.")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-PhishVision-Tool/1.0.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key

        try:
            resp = requests.post(
                f"{portal}/phishvision/scan",
                json={"url": normalized_url},
                headers=headers,
                timeout=self.timeout,
                allow_redirects=False,
            )
            if resp.status_code in (301, 302, 307, 308):
                return "Error: Gateway redirected. Aborting to protect credentials."
            resp.raise_for_status()
            data = resp.json()
            return str(data)
        except Exception as e:
            return f"PhishVision execution error: {str(e)}"

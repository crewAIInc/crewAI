import json
import os
from typing import Any, Optional, Type
from urllib.parse import urlparse
import requests

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class PhishVisionToolSchema(BaseModel):
    """Input schema for the PhishVision threat auditing tool."""

    target: str = Field(
        description="The target URL or domain to audit for cybersecurity threats."
    )


class PhishVisionTool(BaseTool):
    """Pre-flight cybersecurity oracle tool for autonomous agents.

    Audits URLs and domains in real-time for zero-day credential harvesting,
    typosquatting, brand impersonation, and malicious crypto smart contract drainers
    before autonomous agent interaction.
    """

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

    def _run(self, target: str) -> str:
        """Execute the security audit request against the configured PhishVision portal.

        Args:
            target: Target URL or domain to inspect for phishing and drainer threats.

        Returns:
            JSON string containing threat score, verdict, and forensic analysis.

        Raises:
            ValueError: If portal_url does not use HTTPS.
        """
        normalized_url = self._normalize_url(target)
        portal = (self.portal_url or "https://opticparse-api.onrender.com").strip().rstrip("/")
        api_key = (self.api_key or "").strip()

        if not portal.lower().startswith("https://"):
            raise ValueError("portal_url must use a secure HTTPS endpoint to prevent cleartext data transmission.")

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
            if 300 <= resp.status_code < 400:
                return "Error: Gateway redirected. Aborting to protect credentials."
            resp.raise_for_status()
            data = resp.json()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"PhishVision execution error: {str(e)}"

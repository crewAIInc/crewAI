"""CrewAI tools for Darkmoon, a self-hosted autonomous AI penetration testing platform.

The tools talk to the Darkmoon Dashboard API of a Darkmoon instance that you
operate yourself (there is no public hosted endpoint). They log in with
``POST /api/v1/auth/login`` and use the returned JWT as a bearer token.

Only run assessments against systems you own or are explicitly authorised to
test. Findings can include false positives and must be reviewed by a human.
"""

import json
import os
import time
from typing import Any
from urllib.parse import quote

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr
import requests


_TERMINAL_EVENTS = frozenset({"run_completed", "run_error"})
_REQUEST_TIMEOUT = 60


class DarkmoonToolError(Exception):
    """Raised when the Darkmoon Dashboard API cannot be reached or rejects a call."""


class _DarkmoonConnectionMixin(BaseTool):
    """Connection settings and a minimal authenticated client shared by every tool.

    ``base_url``, ``username`` and ``password`` fall back to the
    ``DARKMOON_BASE_URL``, ``DARKMOON_USERNAME`` and ``DARKMOON_PASSWORD``
    environment variables.
    """

    base_url: str | None = Field(
        default=None,
        description="Darkmoon Dashboard API base URL. Defaults to DARKMOON_BASE_URL.",
    )
    username: str | None = Field(
        default=None,
        description="Darkmoon dashboard username. Defaults to DARKMOON_USERNAME.",
    )
    password: str | None = Field(
        default=None,
        repr=False,
        description="Darkmoon dashboard password. Defaults to DARKMOON_PASSWORD.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="DARKMOON_BASE_URL",
                description="Base URL of your self-hosted Darkmoon Dashboard API, e.g. http://localhost:8000",
                required=True,
            ),
            EnvVar(
                name="DARKMOON_USERNAME",
                description="Darkmoon dashboard username",
                required=True,
            ),
            EnvVar(
                name="DARKMOON_PASSWORD",
                description="Darkmoon dashboard password",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    _token: str | None = PrivateAttr(default=None)

    def _settings(self) -> tuple[str, str, str]:
        base_url = self.base_url or os.environ.get("DARKMOON_BASE_URL")
        username = self.username or os.environ.get("DARKMOON_USERNAME")
        password = self.password or os.environ.get("DARKMOON_PASSWORD")
        missing = [
            name
            for name, value in (
                ("DARKMOON_BASE_URL", base_url),
                ("DARKMOON_USERNAME", username),
                ("DARKMOON_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise DarkmoonToolError(
                f"Missing Darkmoon connection settings: {', '.join(missing)}"
            )
        return str(base_url).rstrip("/"), str(username), str(password)

    def _call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        authenticated: bool = True,
    ) -> Any:
        base_url, _, _ = self._settings()
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._login()}"
        try:
            response = requests.request(
                method,
                f"{base_url}{path}",
                headers=headers,
                json=body,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise DarkmoonToolError(f"Request to Darkmoon failed: {exc}") from exc
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise DarkmoonToolError(
                f"Darkmoon API error {response.status_code}: "
                f"{detail if isinstance(detail, str) and detail else 'request failed'}"
            )
        return payload

    def _login(self) -> str:
        if self._token:
            return self._token
        _, username, password = self._settings()
        payload = self._call(
            "POST",
            "/api/v1/auth/login",
            {"username": username, "password": password},
            authenticated=False,
        )
        token = payload.get("token") if isinstance(payload, dict) else None
        if not token:
            raise DarkmoonToolError("Darkmoon login did not return a token")
        self._token = str(token)
        return self._token

    def _list_campaigns(self) -> list[dict[str, Any]]:
        payload = self._call("GET", "/api/v1/campaigns")
        data = payload.get("data") if isinstance(payload, dict) else None
        return data if isinstance(data, list) else []

    def _get_findings(self, campaign_id: str) -> dict[str, Any]:
        payload = self._call(
            "GET",
            f"/api/v1/vulnerabilities?campaign_id={quote(campaign_id, safe='')}",
        )
        body = payload if isinstance(payload, dict) else {}
        return {
            "campaign_id": campaign_id,
            "total": body.get("total") or 0,
            "stats": body.get("stats") or {},
            "findings": body.get("data") or [],
        }


class DarkmoonRunPentestInput(BaseModel):
    """Input schema for DarkmoonRunPentestTool."""

    target: str = Field(
        ...,
        description="Host, URL or scope to assess. Only targets you are authorised to test.",
    )
    wait_for_completion: bool = Field(
        default=True,
        description="Wait for the run to finish and return its findings. "
        "If false, return the run id immediately.",
    )
    program: str | None = Field(
        default=None, description="Optional program name or rules-of-engagement note."
    )
    focus: str | None = Field(
        default=None,
        description='Optional comma separated focus areas, e.g. "auth, injection".',
    )
    severity: str | None = Field(
        default=None, description="Optional minimum severity to report."
    )
    timeout_seconds: int = Field(
        default=1800,
        ge=1,
        description="Maximum time to wait for the run when wait_for_completion is true.",
    )
    poll_interval_seconds: float = Field(
        default=5.0, gt=0, description="Seconds between run status checks."
    )


class DarkmoonRunPentestTool(_DarkmoonConnectionMixin):
    name: str = "Darkmoon Run Pentest"
    description: str = (
        "Start an autonomous Darkmoon penetration test against one authorised target "
        "(host, URL or scope) and return the findings with severity statistics. "
        "Findings may contain false positives and need human review."
    )
    args_schema: type[BaseModel] = DarkmoonRunPentestInput

    def _run(
        self,
        target: str,
        wait_for_completion: bool = True,
        program: str | None = None,
        focus: str | None = None,
        severity: str | None = None,
        timeout_seconds: int = 1800,
        poll_interval_seconds: float = 5.0,
        **_: Any,
    ) -> str:
        target = (target or "").strip()
        if not target:
            raise DarkmoonToolError(
                "A target is required (a host, URL or scope you are authorised to test)."
            )

        params: dict[str, Any] = {"target": target}
        if program and program.strip():
            params["program"] = program.strip()
        areas = [part.strip() for part in (focus or "").split(",") if part.strip()]
        if areas:
            params["focus"] = areas
        if severity and severity.strip():
            params["severity"] = severity.strip()

        known_ids = {campaign.get("id") for campaign in self._list_campaigns()}
        handle = self._call("POST", "/api/v1/run/campaign", params)
        run_id = handle.get("run_id") if isinstance(handle, dict) else None
        if not run_id:
            raise DarkmoonToolError("Darkmoon did not return a run id")

        if not wait_for_completion:
            return json.dumps({"status": "started", "run_id": run_id, "target": target})

        timed_out = self._wait_for_run(run_id, timeout_seconds, poll_interval_seconds)
        campaign = self._resolve_campaign(known_ids, target)
        result: dict[str, Any] = {
            "run_id": run_id,
            "campaign_id": campaign.get("id") if campaign else None,
            "timed_out": timed_out,
            "total": 0,
            "stats": {},
            "findings": [],
        }
        if campaign and campaign.get("id"):
            findings = self._get_findings(str(campaign["id"]))
            result.update(
                total=findings["total"],
                stats=findings["stats"],
                findings=findings["findings"],
            )
        return json.dumps(result)

    def _wait_for_run(self, run_id: str, timeout_seconds: int, poll: float) -> bool:
        """Poll the run log until a terminal event. Returns True on timeout."""
        deadline = time.monotonic() + timeout_seconds
        path = f"/api/v1/run/logs/{quote(run_id, safe='')}"
        while True:
            try:
                payload = self._call("GET", path)
            except DarkmoonToolError as exc:
                # The log file does not exist until the run writes its first event.
                if "404" not in str(exc):
                    raise
                payload = {}
            events = payload.get("data") if isinstance(payload, dict) else None
            if any(e.get("type") in _TERMINAL_EVENTS for e in events or []):
                return False
            if time.monotonic() >= deadline:
                return True
            time.sleep(poll)

    def _resolve_campaign(
        self, known_ids: set[Any], target: str
    ) -> dict[str, Any] | None:
        """Find the campaign minted by the run (the trigger returns a run id only).

        Only campaigns that did not exist before the run are considered, so a
        stale campaign is never reported as the result of this run.
        """
        fresh = [c for c in self._list_campaigns() if c.get("id") not in known_ids]
        if not fresh:
            return None
        fresh.sort(key=lambda c: str(c.get("date") or ""), reverse=True)
        host = target.lower()
        for campaign in fresh:
            if host in str(campaign.get("id", "")).lower():
                return campaign
        return fresh[0]


class DarkmoonGetFindingsInput(BaseModel):
    """Input schema for DarkmoonGetFindingsTool."""

    campaign_id: str = Field(
        ..., description="Darkmoon campaign id, e.g. camp_20260922_abc123."
    )


class DarkmoonGetFindingsTool(_DarkmoonConnectionMixin):
    name: str = "Darkmoon Get Findings"
    description: str = (
        "Return the vulnerabilities and severity statistics Darkmoon recorded for a "
        "campaign id. Findings may contain false positives and need human review."
    )
    args_schema: type[BaseModel] = DarkmoonGetFindingsInput

    def _run(self, campaign_id: str, **_: Any) -> str:
        campaign_id = (campaign_id or "").strip()
        if not campaign_id:
            raise DarkmoonToolError("A campaign id is required.")
        return json.dumps(self._get_findings(campaign_id))


class DarkmoonListCampaignsInput(BaseModel):
    """Input schema for DarkmoonListCampaignsTool (no arguments)."""


class DarkmoonListCampaignsTool(_DarkmoonConnectionMixin):
    name: str = "Darkmoon List Campaigns"
    description: str = (
        "List the Darkmoon campaigns visible to the authenticated dashboard user, "
        "with their ids."
    )
    args_schema: type[BaseModel] = DarkmoonListCampaignsInput

    def _run(self, **_: Any) -> str:
        campaigns = self._list_campaigns()
        return json.dumps({"total": len(campaigns), "campaigns": campaigns})

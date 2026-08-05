"""CrewAI tool for current provider and service status from OutageDeck."""

import asyncio
import json
import re
from typing import Any, Literal
from urllib.parse import urlencode

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator, model_validator
import requests
from typing_extensions import Self

from crewai_tools.security.safe_requests import safe_get


OutageDeckOperation = Literal["provider_status", "list_incidents", "service_status"]
IncidentState = Literal["active", "resolved"]
IncidentSeverity = Literal["minor", "major", "critical", "maintenance"]

_API_BASE_URL = "https://outagedeck.com/api/v1"
_SITE_BASE_URL = "https://outagedeck.com"
_USER_AGENT = "CrewAI-OutageDeck/1.0"
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ATTRIBUTION = {
    "utm_source": "crewai",
    "utm_medium": "integration",
    "utm_campaign": "crewai_tool",
}


class OutageDeckStatusInput(BaseModel):
    """Input for OutageDeckStatusTool."""

    operation: OutageDeckOperation = Field(
        description=(
            "Operation to run: provider_status, list_incidents, or service_status."
        )
    )
    slug: str | None = Field(
        default=None,
        description=(
            "Provider or service slug required by provider_status and service_status, "
            "for example github or github-actions."
        ),
    )
    provider: str | None = Field(
        default=None,
        description="Optional provider slug filter for list_incidents.",
    )
    state: IncidentState | None = Field(
        default=None,
        description="Optional incident lifecycle filter: active or resolved.",
    )
    severity: IncidentSeverity | None = Field(
        default=None,
        description="Optional severity filter: minor, major, critical, or maintenance.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="1-based incident result page.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Incidents per page from 1 through 100.",
    )

    @field_validator("slug", "provider")
    @classmethod
    def validate_slug(cls, value: str | None) -> str | None:
        """Normalize a slug and reject values that could alter an API path."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _SLUG_PATTERN.fullmatch(normalized):
            raise ValueError(
                "slugs must contain only lowercase letters, numbers, and single hyphens"
            )
        return normalized

    @model_validator(mode="after")
    def require_operation_slug(self) -> Self:
        """Require a path slug for operations that address one resource."""
        if (
            self.operation in {"provider_status", "service_status"}
            and self.slug is None
        ):
            raise ValueError(f"slug is required for {self.operation}")
        return self


class OutageDeckStatusTool(BaseTool):
    """Check live infrastructure status and incidents through OutageDeck.

    OutageDeck normalizes provider status feeds into a public, read-only API.
    This tool requires no API key and uses a fixed production origin.
    """

    name: str = "OutageDeck Status"
    description: str = (
        "Check current infrastructure-provider and service status with OutageDeck. "
        "Use provider_status with a provider slug such as github or openai; "
        "use list_incidents for active or historical incidents with optional provider, "
        "state, and severity filters; use service_status with a service slug such as "
        "github-actions or openai-api. The tool is read-only and requires no API key."
    )
    args_schema: type[BaseModel] = OutageDeckStatusInput
    timeout: int = Field(
        default=20,
        gt=0,
        le=120,
        description="HTTP request timeout in seconds.",
    )

    def _run(
        self,
        operation: OutageDeckOperation,
        slug: str | None = None,
        provider: str | None = None,
        state: IncidentState | None = None,
        severity: IncidentSeverity | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> str:
        """Run one OutageDeck status operation and return structured JSON."""
        request = OutageDeckStatusInput(
            operation=operation,
            slug=slug,
            provider=provider,
            state=state,
            severity=severity,
            page=page,
            limit=limit,
        )
        return self._execute(request)

    async def _arun(
        self,
        operation: OutageDeckOperation,
        slug: str | None = None,
        provider: str | None = None,
        state: IncidentState | None = None,
        severity: IncidentSeverity | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> str:
        """Run one OutageDeck status operation without blocking the event loop."""
        request = OutageDeckStatusInput(
            operation=operation,
            slug=slug,
            provider=provider,
            state=state,
            severity=severity,
            page=page,
            limit=limit,
        )
        return await asyncio.to_thread(self._execute, request)

    def _execute(self, request: OutageDeckStatusInput) -> str:
        url, params, resource = self._request_details(request)
        try:
            response = safe_get(
                url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _USER_AGENT,
                },
                timeout=self.timeout,
                max_redirects=0,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            return self._failure(self._http_error(error, resource))
        except requests.RequestException as error:
            return self._failure(f"Could not reach OutageDeck: {error}")
        except ValueError as error:
            return self._failure(f"OutageDeck request rejected: {error}")

        try:
            parsed = self._parse_response(request.operation, response.json())
        except ValueError as error:
            return self._failure(f"OutageDeck returned invalid JSON: {error}")
        if "error" in parsed:
            return self._failure(str(parsed["error"]))
        return self._success(parsed)

    @staticmethod
    def _request_details(
        request: OutageDeckStatusInput,
    ) -> tuple[str, dict[str, str | int], str]:
        if request.operation == "provider_status":
            return (
                f"{_API_BASE_URL}/providers/{request.slug}",
                {},
                f"provider '{request.slug}'",
            )
        if request.operation == "service_status":
            return (
                f"{_API_BASE_URL}/services/{request.slug}",
                {},
                f"service '{request.slug}'",
            )

        params: dict[str, str | int] = {
            "page": request.page,
            "limit": request.limit,
        }
        if request.provider is not None:
            params["provider"] = request.provider
        if request.state is not None:
            params["state"] = request.state
        if request.severity is not None:
            params["severity"] = request.severity
        return f"{_API_BASE_URL}/incidents", params, "the requested incidents"

    @classmethod
    def _parse_response(
        cls, operation: OutageDeckOperation, payload: Any
    ) -> dict[str, Any]:
        if operation == "provider_status":
            return cls._parse_provider(payload)
        if operation == "service_status":
            return cls._parse_service(payload)
        return cls._parse_incidents(payload)

    @staticmethod
    def _data(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None

    @classmethod
    def _parse_provider(cls, payload: Any) -> dict[str, Any]:
        data = cls._data(payload)
        if data is None:
            return {"error": "OutageDeck returned an unexpected provider response."}
        status = data.get("currentStatus")
        slug = data.get("slug")
        if not isinstance(status, dict) or not isinstance(slug, str):
            return {"error": "OutageDeck returned an unexpected provider response."}

        services = data.get("services")
        incidents = data.get("activeIncidents")
        return {
            "operation": "provider_status",
            "provider": {"slug": slug, "name": data.get("name")},
            "status": {
                "code": status.get("code"),
                "label": status.get("label"),
                "headline": status.get("headline"),
                "summary": status.get("summary"),
                "captured_at": status.get("capturedAt"),
            },
            "counts": data.get("counts")
            if isinstance(data.get("counts"), dict)
            else {},
            "services": [
                {
                    "slug": service.get("slug"),
                    "name": service.get("name"),
                    "category": service.get("category"),
                    "status": service.get("status"),
                    "summary": service.get("summary"),
                }
                for service in services
                if isinstance(service, dict)
            ]
            if isinstance(services, list)
            else [],
            "active_incidents": cls._normalized_incidents(incidents, limit=10),
            "outagedeck_url": cls._attributed_url(f"/providers/{slug}"),
        }

    @classmethod
    def _parse_incidents(cls, payload: Any) -> dict[str, Any]:
        data = cls._data(payload)
        if data is None or not isinstance(data.get("incidents"), list):
            return {"error": "OutageDeck returned an unexpected incident response."}
        incidents = cls._normalized_incidents(data["incidents"])
        return {
            "operation": "list_incidents",
            "count": data.get("count", len(incidents)),
            "page": data.get("page"),
            "total_pages": data.get("totalPages"),
            "total_incidents": data.get("totalIncidents"),
            "incidents": incidents,
            "outagedeck_url": cls._attributed_url("/incidents"),
        }

    @classmethod
    def _parse_service(cls, payload: Any) -> dict[str, Any]:
        data = cls._data(payload)
        if data is None:
            return {"error": "OutageDeck returned an unexpected service response."}
        slug = data.get("slug")
        provider = data.get("provider")
        if not isinstance(slug, str) or not isinstance(provider, dict):
            return {"error": "OutageDeck returned an unexpected service response."}
        return {
            "operation": "service_status",
            "service": {
                "slug": slug,
                "name": data.get("name"),
                "category": data.get("category"),
                "status": data.get("status"),
                "summary": data.get("summary"),
            },
            "provider": {"slug": provider.get("slug"), "name": provider.get("name")},
            "counts": data.get("counts")
            if isinstance(data.get("counts"), dict)
            else {},
            "recent_incidents": cls._normalized_incidents(
                data.get("incidents"), limit=10
            ),
            "outagedeck_url": cls._attributed_url(f"/services/{slug}"),
        }

    @classmethod
    def _normalized_incidents(
        cls, incidents: Any, limit: int | None = None
    ) -> list[dict[str, Any]]:
        if not isinstance(incidents, list):
            return []
        normalized = [
            incident
            for item in incidents
            if (incident := cls._incident(item)) is not None
        ]
        return normalized[:limit] if limit is not None else normalized

    @classmethod
    def _incident(cls, incident: Any) -> dict[str, Any] | None:
        if not isinstance(incident, dict):
            return None
        provider = incident.get("provider")
        affected = incident.get("affectedServices")
        slug = incident.get("slug")
        result = {
            "slug": slug,
            "title": incident.get("title"),
            "summary": incident.get("summary"),
            "status": incident.get("status"),
            "severity": incident.get("severity"),
            "started_at": incident.get("startedAt"),
            "updated_at": incident.get("updatedAt"),
            "resolved_at": incident.get("resolvedAt"),
            "provider": {
                "slug": provider.get("slug"),
                "name": provider.get("name"),
            }
            if isinstance(provider, dict)
            else None,
            "affected_services": [
                {"slug": service.get("slug"), "name": service.get("name")}
                for service in affected
                if isinstance(service, dict)
            ]
            if isinstance(affected, list)
            else [],
        }
        if isinstance(slug, str) and _SLUG_PATTERN.fullmatch(slug):
            result["outagedeck_url"] = cls._attributed_url(f"/incidents/{slug}")
        return result

    @staticmethod
    def _attributed_url(path: str) -> str:
        return f"{_SITE_BASE_URL}{path}?{urlencode(_ATTRIBUTION)}"

    @staticmethod
    def _http_error(error: requests.HTTPError, resource: str) -> str:
        status = error.response.status_code if error.response is not None else None
        if status == 404:
            return f"OutageDeck could not find {resource}."
        if status == 429:
            return "OutageDeck rate limit exceeded. Try again later."
        if status in {401, 403}:
            return "OutageDeck rejected the request."
        return f"OutageDeck API error: HTTP {status or 'unknown'}."

    @staticmethod
    def _success(result: dict[str, Any]) -> str:
        return json.dumps({"success": True, **result}, separators=(",", ":"))

    @staticmethod
    def _failure(message: str) -> str:
        return json.dumps({"success": False, "error": message}, separators=(",", ":"))

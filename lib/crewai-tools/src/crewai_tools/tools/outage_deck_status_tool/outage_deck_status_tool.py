"""Tool that checks infrastructure provider status via OutageDeck public API."""

import json
import re
from typing import Any, Literal, cast

from crewai.tools import BaseTool
import httpx
from pydantic import BaseModel, Field, field_validator


OUTAGE_DECK_BASE_URL = "https://outagedeck.com/api/v1"

# Slug pattern: lowercase letters, digits, and hyphens only
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Known valid values for filter parameters
_VALID_LIFECYCLES = {"active", "resolved"}
_VALID_SEVERITIES = {"minor", "major", "critical"}
_VALID_ACTIONS = {"provider_status", "list_incidents", "service_status"}

ActionType = Literal["provider_status", "list_incidents", "service_status"]


def _validate_slug(value: str, field_name: str) -> str:
    """Validate that a slug matches the expected format.

    Args:
        value: The slug string to validate.
        field_name: Name of the field being validated, used in error messages.

    Returns:
        The validated slug.

    Raises:
        ValueError: If the slug is empty or has an invalid format.
    """
    if not value:
        raise ValueError(f"{field_name} must not be empty.")
    if not _SLUG_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, digits, and hyphens, "
            f"and must not start or end with a hyphen. Got: {value!r}"
        )
    return value


class OutageDeckStatusToolSchema(BaseModel):
    """Input for OutageDeckStatusTool."""

    action: ActionType = Field(
        ...,
        description=(
            "Which status check to perform. Use 'provider_status' to get the overall "
            "status of a provider with all its services and active incidents. Use "
            "'list_incidents' to browse paginated incidents with optional filters. Use "
            "'service_status' to get the status of a specific service with recent "
            "incident context."
        ),
    )
    provider_slug: str | None = Field(
        default=None,
        description=(
            "Provider slug for 'provider_status' or as a filter for 'list_incidents'. "
            "Examples: 'aws', 'openai', 'github', 'vercel'. "
            "Required when action is 'provider_status'."
        ),
    )
    service_slug: str | None = Field(
        default=None,
        description=(
            "Service slug for 'service_status'. Examples: 'aws-ec2', 'openai-api', "
            "'github-actions'. Required when action is 'service_status'."
        ),
    )
    lifecycle: str | None = Field(
        default=None,
        description=(
            "Optional filter for 'list_incidents'. Only return incidents in this "
            "lifecycle state. Valid values: 'active', 'resolved'."
        ),
    )
    severity: str | None = Field(
        default=None,
        description=(
            "Optional filter for 'list_incidents'. Only return incidents of this "
            "severity. Valid values: 'minor', 'major', 'critical'."
        ),
    )
    page: int = Field(
        default=1,
        ge=1,
        description=(
            "Page number for 'list_incidents' pagination. Defaults to 1. "
            "Each page contains up to 25 incidents."
        ),
    )

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        if value not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}, got {value!r}."
            )
        return value

    @field_validator("provider_slug")
    @classmethod
    def _validate_provider_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_slug(value, "provider_slug")

    @field_validator("service_slug")
    @classmethod
    def _validate_service_slug(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_slug(value, "service_slug")

    @field_validator("lifecycle")
    @classmethod
    def _validate_lifecycle(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _VALID_LIFECYCLES:
            raise ValueError(
                f"lifecycle must be one of {sorted(_VALID_LIFECYCLES)}, got {value!r}."
            )
        return value

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {value!r}."
            )
        return value


class OutageDeckStatusTool(BaseTool):
    """Check infrastructure provider and service status using the OutageDeck public API.

    Agents can use this tool to verify whether a third-party service is experiencing
    an outage before debugging their own code, or to pull recent incident history for
    context about reliability issues.

    The OutageDeck public API is read-only and requires no API key or account.
    Three operations are supported:
    - provider_status: overall status of a provider plus its services and active incidents
    - list_incidents: paginated incident list with optional provider/lifecycle/severity filters
    - service_status: status of a single service with limited recent incident context
    """

    name: str = "OutageDeck status check"
    description: str = (
        "Check the current status of infrastructure providers and services using "
        "the OutageDeck public status API. Use this when you need to know whether "
        "a third-party service (such as a cloud provider, API, or SaaS platform) "
        "is currently experiencing an outage, degraded performance, or has recent "
        "incident history. No API key is required. Three operations are available: "
        "'provider_status' returns the overall status of a provider with its services "
        "and active incidents; 'list_incidents' returns paginated incidents with "
        "optional filters by provider, lifecycle, and severity; 'service_status' "
        "returns the status of a specific service with recent incident context."
    )
    args_schema: type[BaseModel] = OutageDeckStatusToolSchema

    def _run(
        self,
        action: str,
        provider_slug: str | None = None,
        service_slug: str | None = None,
        lifecycle: str | None = None,
        severity: str | None = None,
        page: int = 1,
    ) -> str:
        """Execute a status check against the OutageDeck public API.

        Args:
            action: Which operation to perform.
            provider_slug: Provider slug for provider_status or list_incidents filter.
            service_slug: Service slug for service_status.
            lifecycle: Optional lifecycle filter for list_incidents.
            severity: Optional severity filter for list_incidents.
            page: Page number for list_incidents pagination.

        Returns:
            JSON string with the status data returned by the API.
        """
        data = self._fetch(action, provider_slug, service_slug, lifecycle, severity, page)
        return json.dumps(data, ensure_ascii=False, indent=2)

    async def _arun(
        self,
        action: str,
        provider_slug: str | None = None,
        service_slug: str | None = None,
        lifecycle: str | None = None,
        severity: str | None = None,
        page: int = 1,
    ) -> str:
        """Asynchronously execute a status check against the OutageDeck public API.

        Args:
            action: Which operation to perform.
            provider_slug: Provider slug for provider_status or list_incidents filter.
            service_slug: Service slug for service_status.
            lifecycle: Optional lifecycle filter for list_incidents.
            severity: Optional severity filter for list_incidents.
            page: Page number for list_incidents pagination.

        Returns:
            JSON string with the status data returned by the API.
        """
        data = await self._async_fetch(
            action, provider_slug, service_slug, lifecycle, severity, page
        )
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _fetch(
        self,
        action: str,
        provider_slug: str | None,
        service_slug: str | None,
        lifecycle: str | None,
        severity: str | None,
        page: int,
    ) -> dict[str, Any]:
        """Perform the HTTP request synchronously and return parsed JSON.

        Args:
            action: Which operation to perform.
            provider_slug: Provider slug.
            service_slug: Service slug.
            lifecycle: Lifecycle filter.
            severity: Severity filter.
            page: Page number.

        Returns:
            Parsed JSON response from the API.

        Raises:
            ValueError: If required parameters are missing for the chosen action.
            httpx.HTTPError: If the HTTP request fails.
        """
        url, params = self._build_request(
            action, provider_slug, service_slug, lifecycle, severity, page
        )
        response = httpx.get(url, params=params, timeout=15.0)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def _async_fetch(
        self,
        action: str,
        provider_slug: str | None,
        service_slug: str | None,
        lifecycle: str | None,
        severity: str | None,
        page: int,
    ) -> dict[str, Any]:
        """Perform the HTTP request asynchronously and return parsed JSON.

        Args:
            action: Which operation to perform.
            provider_slug: Provider slug.
            service_slug: Service slug.
            lifecycle: Lifecycle filter.
            severity: Severity filter.
            page: Page number.

        Returns:
            Parsed JSON response from the API.

        Raises:
            ValueError: If required parameters are missing for the chosen action.
            httpx.HTTPError: If the HTTP request fails.
        """
        url, params = self._build_request(
            action, provider_slug, service_slug, lifecycle, severity, page
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return cast(dict[str, Any], response.json())

    @staticmethod
    def _build_request(
        action: str,
        provider_slug: str | None,
        service_slug: str | None,
        lifecycle: str | None,
        severity: str | None,
        page: int,
    ) -> tuple[str, dict[str, Any]]:
        """Build the URL and query parameters for the API request.

        Args:
            action: Which operation to perform.
            provider_slug: Provider slug.
            service_slug: Service slug.
            lifecycle: Lifecycle filter.
            severity: Severity filter.
            page: Page number.

        Returns:
            A tuple of (url, params_dict).

        Raises:
            ValueError: If required parameters are missing for the chosen action.
        """
        params: dict[str, Any] = {}

        if action == "provider_status":
            if not provider_slug:
                raise ValueError(
                    "provider_slug is required when action is 'provider_status'."
                )
            url = f"{OUTAGE_DECK_BASE_URL}/providers/{provider_slug}"

        elif action == "service_status":
            if not service_slug:
                raise ValueError(
                    "service_slug is required when action is 'service_status'."
                )
            url = f"{OUTAGE_DECK_BASE_URL}/services/{service_slug}"

        elif action == "list_incidents":
            url = f"{OUTAGE_DECK_BASE_URL}/incidents"
            if provider_slug:
                params["provider"] = provider_slug
            if lifecycle:
                params["lifecycle"] = lifecycle
            if severity:
                params["severity"] = severity
            if page > 1:
                params["page"] = page

        else:
            raise ValueError(
                f"Unknown action: {action!r}. "
                f"Must be one of {sorted(_VALID_ACTIONS)}."
            )

        return url, params

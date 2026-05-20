"""CrewAI+ API client — shared by both crewai and crewai-cli."""

from __future__ import annotations

import os
from typing import Any, Final, Literal, TypedDict, cast
from urllib.parse import urljoin

import httpx
from typing_extensions import NotRequired

from crewai_core.constants import DEFAULT_CREWAI_ENTERPRISE_URL
from crewai_core.settings import Settings
from crewai_core.version import get_crewai_version


HttpMethod = Literal["GET", "POST", "PATCH", "DELETE"]


class AvailableExport(TypedDict):
    name: str


class EnvVarEntry(TypedDict):
    name: str
    description: str
    required: bool
    default: str | None


class ToolMetadata(TypedDict):
    name: str
    module: str
    humanized_name: str
    description: str
    run_params_schema: dict[str, Any]
    init_params_schema: dict[str, Any]
    env_vars: list[EnvVarEntry]


class ToolsMetadataPayload(TypedDict):
    package: str
    tools: list[ToolMetadata] | None


class PublishToolPayload(TypedDict):
    handle: str
    public: bool
    version: str
    file: str
    description: str | None
    available_exports: list[AvailableExport] | None
    tools_metadata: ToolsMetadataPayload | None


class CrewDeploymentSpec(TypedDict):
    name: str
    repo_clone_url: str
    env: dict[str, str]


class CreateCrewPayload(TypedDict):
    deploy: CrewDeploymentSpec


class _WithUserIdentifier(TypedDict):
    user_identifier: NotRequired[str]


class LoginPayload(_WithUserIdentifier):
    pass


class TraceExecutionContext(TypedDict):
    crew_fingerprint: str | None
    crew_name: str | None
    flow_name: str | None
    crewai_version: str
    privacy_level: str


class TraceExecutionMetadata(TypedDict):
    expected_duration_estimate: int
    agent_count: int
    task_count: int
    flow_method_count: int
    execution_started_at: str


class TraceBatchInitPayload(_WithUserIdentifier):
    trace_id: str
    execution_type: str
    execution_context: TraceExecutionContext
    execution_metadata: TraceExecutionMetadata
    ephemeral_trace_id: NotRequired[str]


class TraceBatchMetadata(TypedDict):
    events_count: int
    batch_sequence: int
    is_final_batch: bool


class TraceEventsPayload(TypedDict):
    events: list[dict[str, Any]]
    batch_metadata: TraceBatchMetadata


class TraceFinalizePayload(TypedDict):
    status: Literal["completed"]
    duration_ms: float | None
    final_event_count: int


class TraceFailedPayload(TypedDict):
    status: Literal["failed"]
    failure_reason: str


Headers = TypedDict(
    "Headers",
    {
        "Content-Type": str,
        "User-Agent": str,
        "X-Crewai-Version": str,
        "Authorization": NotRequired[str],
        "X-Crewai-Organization-Id": NotRequired[str],
    },
)


class RequestKwargs(TypedDict):
    headers: dict[str, str]
    json: NotRequired[Any]
    params: NotRequired[dict[str, str]]
    timeout: NotRequired[float]


class PlusAPI:
    """Client for working with the CrewAI+ API."""

    TOOLS_RESOURCE: Final = "/crewai_plus/api/v1/tools"
    ORGANIZATIONS_RESOURCE: Final = "/crewai_plus/api/v1/me/organizations"
    CREWS_RESOURCE: Final = "/crewai_plus/api/v1/crews"
    AGENTS_RESOURCE: Final = "/crewai_plus/api/v1/agents"
    TRACING_RESOURCE: Final = "/crewai_plus/api/v1/tracing"
    EPHEMERAL_TRACING_RESOURCE: Final = "/crewai_plus/api/v1/tracing/ephemeral"
    INTEGRATIONS_RESOURCE: Final = "/crewai_plus/api/v1/integrations"

    def __init__(self, api_key: str | None = None) -> None:
        version = get_crewai_version()
        self.api_key = api_key
        self.headers: Headers = {
            "Content-Type": "application/json",
            "User-Agent": f"CrewAI-CLI/{version}",
            "X-Crewai-Version": version,
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        settings = Settings()
        if settings.org_uuid:
            self.headers["X-Crewai-Organization-Id"] = settings.org_uuid

        self.base_url = (
            os.getenv("CREWAI_PLUS_URL")
            or str(settings.enterprise_base_url)
            or DEFAULT_CREWAI_ENTERPRISE_URL
        )

    def _make_request(
        self,
        method: HttpMethod,
        endpoint: str,
        *,
        json: Any = None,
        params: dict[str, str] | None = None,
        timeout: float | None = None,
        verify: bool = True,
    ) -> httpx.Response:
        url = urljoin(self.base_url, endpoint)
        request_kwargs: RequestKwargs = {"headers": cast(dict[str, str], self.headers)}
        if json is not None:
            request_kwargs["json"] = json
        if params is not None:
            request_kwargs["params"] = params
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        with httpx.Client(trust_env=False, verify=verify) as client:
            return client.request(method, url, **request_kwargs)

    def login_to_tool_repository(
        self, user_identifier: str | None = None
    ) -> httpx.Response:
        payload: LoginPayload = {}
        if user_identifier:
            payload["user_identifier"] = user_identifier
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}/login", json=payload)

    def get_tool(self, handle: str) -> httpx.Response:
        return self._make_request("GET", f"{self.TOOLS_RESOURCE}/{handle}")

    async def get_agent(self, handle: str) -> httpx.Response:
        url = urljoin(self.base_url, f"{self.AGENTS_RESOURCE}/{handle}")
        async with httpx.AsyncClient() as client:
            return await client.get(url, headers=cast(dict[str, str], self.headers))

    def publish_tool(
        self,
        handle: str,
        is_public: bool,
        version: str,
        description: str | None,
        encoded_file: str,
        available_exports: list[AvailableExport] | None = None,
        tools_metadata: list[ToolMetadata] | None = None,
    ) -> httpx.Response:
        params: PublishToolPayload = {
            "handle": handle,
            "public": is_public,
            "version": version,
            "file": encoded_file,
            "description": description,
            "available_exports": available_exports,
            "tools_metadata": {"package": handle, "tools": tools_metadata}
            if tools_metadata is not None
            else None,
        }
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}", json=params)

    def deploy_by_name(self, project_name: str) -> httpx.Response:
        return self._make_request(
            "POST", f"{self.CREWS_RESOURCE}/by-name/{project_name}/deploy"
        )

    def deploy_by_uuid(self, uuid: str) -> httpx.Response:
        return self._make_request("POST", f"{self.CREWS_RESOURCE}/{uuid}/deploy")

    def crew_status_by_name(self, project_name: str) -> httpx.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/by-name/{project_name}/status"
        )

    def crew_status_by_uuid(self, uuid: str) -> httpx.Response:
        return self._make_request("GET", f"{self.CREWS_RESOURCE}/{uuid}/status")

    def crew_by_name(
        self, project_name: str, log_type: str = "deployment"
    ) -> httpx.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/by-name/{project_name}/logs/{log_type}"
        )

    def crew_by_uuid(self, uuid: str, log_type: str = "deployment") -> httpx.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/{uuid}/logs/{log_type}"
        )

    def delete_crew_by_name(self, project_name: str) -> httpx.Response:
        return self._make_request(
            "DELETE", f"{self.CREWS_RESOURCE}/by-name/{project_name}"
        )

    def delete_crew_by_uuid(self, uuid: str) -> httpx.Response:
        return self._make_request("DELETE", f"{self.CREWS_RESOURCE}/{uuid}")

    def list_crews(self) -> httpx.Response:
        return self._make_request("GET", self.CREWS_RESOURCE)

    def create_crew(self, payload: CreateCrewPayload) -> httpx.Response:
        return self._make_request("POST", self.CREWS_RESOURCE, json=payload)

    def get_organizations(self) -> httpx.Response:
        return self._make_request("GET", self.ORGANIZATIONS_RESOURCE)

    def initialize_trace_batch(self, payload: TraceBatchInitPayload) -> httpx.Response:
        return self._make_request(
            "POST",
            f"{self.TRACING_RESOURCE}/batches",
            json=payload,
            timeout=30,
        )

    def initialize_ephemeral_trace_batch(
        self, payload: TraceBatchInitPayload
    ) -> httpx.Response:
        return self._make_request(
            "POST",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches",
            json=payload,
        )

    def send_trace_events(
        self, trace_batch_id: str, payload: TraceEventsPayload
    ) -> httpx.Response:
        return self._make_request(
            "POST",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
            timeout=30,
        )

    def send_ephemeral_trace_events(
        self, trace_batch_id: str, payload: TraceEventsPayload
    ) -> httpx.Response:
        return self._make_request(
            "POST",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
            timeout=30,
        )

    def finalize_trace_batch(
        self, trace_batch_id: str, payload: TraceFinalizePayload
    ) -> httpx.Response:
        return self._make_request(
            "PATCH",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
            timeout=30,
        )

    def finalize_ephemeral_trace_batch(
        self, trace_batch_id: str, payload: TraceFinalizePayload
    ) -> httpx.Response:
        return self._make_request(
            "PATCH",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
            timeout=30,
        )

    def mark_trace_batch_as_failed(
        self, trace_batch_id: str, error_message: str
    ) -> httpx.Response:
        payload: TraceFailedPayload = {
            "status": "failed",
            "failure_reason": error_message,
        }
        return self._make_request(
            "PATCH",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}",
            json=payload,
            timeout=30,
        )

    def mark_ephemeral_trace_batch_as_failed(
        self, trace_batch_id: str, error_message: str
    ) -> httpx.Response:
        payload: TraceFailedPayload = {
            "status": "failed",
            "failure_reason": error_message,
        }
        return self._make_request(
            "PATCH",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}",
            json=payload,
            timeout=30,
        )

    def get_mcp_configs(self, slugs: list[str]) -> httpx.Response:
        """Get MCP server configurations for the given slugs."""
        return self._make_request(
            "GET",
            f"{self.INTEGRATIONS_RESOURCE}/mcp_configs",
            params={"slugs": ",".join(slugs)},
            timeout=30,
        )

    def get_triggers(self) -> httpx.Response:
        """Get all available triggers from integrations."""
        return self._make_request("GET", f"{self.INTEGRATIONS_RESOURCE}/apps")

    def get_trigger_payload(self, app_slug: str, trigger_slug: str) -> httpx.Response:
        """Get sample payload for a specific trigger."""
        return self._make_request(
            "GET", f"{self.INTEGRATIONS_RESOURCE}/{app_slug}/{trigger_slug}/payload"
        )

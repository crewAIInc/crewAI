"""CrewAI+ API client — shared by both crewai and crewai-cli."""

from __future__ import annotations

from collections.abc import MutableMapping
import ipaddress
import os
from pathlib import Path
import ssl
from typing import Any, Final, Literal, TypedDict, cast
from urllib.parse import urljoin
import urllib.request

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
    # ``[tool.crewai].project_id``, at the top level as every other deploy
    # request sends it (AMP reads it there on every route); only when set.
    project_id: NotRequired[str]


class _WithUserIdentifier(TypedDict):
    user_identifier: NotRequired[str]


class LoginPayload(_WithUserIdentifier):
    project_id: NotRequired[str]


class TraceExecutionContext(TypedDict):
    crew_fingerprint: str | None
    crew_name: str | None
    flow_name: str | None
    crewai_version: str
    privacy_level: str
    project_id: NotRequired[str | None]


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


# The integrations apps listing takes ~11s on a real organization. Only the read
# waits longer, so an unreachable or saturated service still fails at httpx's 5s.
_TRIGGER_TIMEOUT = httpx.Timeout(5.0, read=60.0)


class RequestKwargs(TypedDict):
    headers: dict[str, str]
    json: NotRequired[Any]
    params: NotRequired[dict[str, str]]
    timeout: NotRequired[float | httpx.Timeout]


def _sets_transport(key: str) -> bool:
    """Whether this variable decides the proxy or CA bundle of AMP requests.

    httpx reads the proxy and CA variables. REQUEST_METHOD counts: urllib drops
    an uppercase HTTP_PROXY when it is set. So does UV_ENV_FILE, for a crewAI
    process started with `uv run`: uv loads that file into it before Python starts.
    """
    return key.lower().endswith("_proxy") or key.upper() in {
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUEST_METHOD",
        "UV_ENV_FILE",
    }


# The proxy and CA variables as the process had them when this module was first
# imported. AMP requests carry the saved login, so they take their proxy and CA
# from here, never from the live environment: a project's .env, loaded later by
# `crewai run` or by `crewai.llm` on import, must not choose where the login
# travels or which CA vouches for AMP. The CLI imports this module at startup,
# before any command reads a .env.
_MACHINE_TRANSPORT_ENV: dict[str, str] = {
    key: value for key, value in os.environ.items() if _sets_transport(key)
}


def restore_machine_transport(environ: MutableMapping[str, str]) -> None:
    """Put the pinned proxy and CA variables back in ``environ``, dropping any set since.

    For a process that loads a project's .env and then starts another crewAI
    process: the child pins the variables it inherits as the machine's.
    """
    for key in [key for key in environ if _sets_transport(key)]:
        del environ[key]
    environ.update(_MACHINE_TRANSPORT_ENV)


def _os_proxies() -> dict[str, str]:
    """The OS proxy settings (macOS System Settings, the Windows registry)."""
    read = getattr(urllib.request, "getproxies_macosx_sysconf", None) or getattr(
        urllib.request, "getproxies_registry", None
    )
    return read() if read else {}


def _machine_proxies() -> dict[str, str]:
    """Proxies by scheme, read from the pinned variables as urllib's getproxies() reads them."""
    proxies: dict[str, str] = {}
    # Two passes, so a lowercase name wins over its uppercase twin.
    for name, value in _MACHINE_TRANSPORT_ENV.items():
        if value and name.lower().endswith("_proxy"):
            proxies[name[:-6].lower()] = value
    # As urllib does (CVE-2016-1000110, "httpoxy"): in a CGI process a client's
    # Proxy header arrives as HTTP_PROXY, so only a lowercase http_proxy counts.
    if "REQUEST_METHOD" in _MACHINE_TRANSPORT_ENV:
        proxies.pop("http", None)
    for name, value in _MACHINE_TRANSPORT_ENV.items():
        if name.endswith("_proxy"):
            if value:
                proxies[name[:-6].lower()] = value
            else:
                proxies.pop(name[:-6].lower(), None)
    return proxies or _os_proxies()


def _no_proxy_pattern(host: str) -> str:
    """The httpx mount pattern for one NO_PROXY entry, as httpx builds it."""
    if "://" in host:
        return host
    try:
        address = ipaddress.ip_address(host.split("/")[0])
    except ValueError:
        return f"all://{host}" if host.lower() == "localhost" else f"all://*{host}"
    return f"all://[{host}]" if address.version == 6 else f"all://{host}"


def _machine_transport(verify: bool) -> dict[str, Any]:
    """httpx.Client arguments for the machine's proxy and CA, as trust_env would read them at startup."""
    ssl_verify: ssl.SSLContext | bool = verify
    if verify and (cafile := _MACHINE_TRANSPORT_ENV.get("SSL_CERT_FILE")):
        ssl_verify = ssl.create_default_context(cafile=cafile)
    elif verify and (capath := _MACHINE_TRANSPORT_ENV.get("SSL_CERT_DIR")):
        ssl_verify = ssl.create_default_context(capath=capath)

    proxies = _machine_proxies()
    mounts: dict[str, httpx.BaseTransport | None] = {}
    for scheme in ("http", "https", "all"):
        if url := proxies.get(scheme):
            mounts[f"{scheme}://"] = httpx.HTTPTransport(
                proxy=url if "://" in url else f"http://{url}",
                verify=ssl_verify,
                trust_env=False,
            )
    for host in (entry.strip() for entry in proxies.get("no", "").split(",")):
        if host == "*":
            mounts = {}
            break
        if host:
            mounts[_no_proxy_pattern(host)] = None
    return {"verify": ssl_verify, "trust_env": False, "mounts": mounts}


class PlusAPI:
    """Client for working with the CrewAI+ API."""

    TOOLS_RESOURCE: Final = "/crewai_plus/api/v1/tools"
    SKILLS_RESOURCE: Final = "/crewai_plus/api/v1/skills"
    ORGANIZATIONS_RESOURCE: Final = "/crewai_plus/api/v1/me/organizations"
    INTEGRATION_TOKEN_RESOURCE: Final = "/crewai_plus/api/v1/me/integration-token"  # noqa: S105 - API path, not a secret
    CREWS_RESOURCE: Final = "/crewai_plus/api/v1/crews"
    AGENTS_RESOURCE: Final = "/crewai_plus/api/v1/agents"
    TRACING_RESOURCE: Final = "/crewai_plus/api/v1/tracing"
    EPHEMERAL_TRACING_RESOURCE: Final = "/crewai_plus/api/v1/tracing/ephemeral"
    INTEGRATIONS_RESOURCE: Final = "/crewai_plus/api/v1/integrations"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        organization_id: str | None = None,
    ) -> None:
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
        if organization_id := organization_id or settings.org_uuid:
            self.headers["X-Crewai-Organization-Id"] = organization_id

        self.base_url = (
            base_url
            or os.getenv("CREWAI_PLUS_URL")
            or settings.enterprise_base_url
            or DEFAULT_CREWAI_ENTERPRISE_URL
        )

    @staticmethod
    def _http_client(verify: bool = True) -> httpx.Client:
        """A client that reaches AMP through the machine's proxy and CA, not a project's .env."""
        return httpx.Client(**_machine_transport(verify))

    def _make_request(
        self,
        method: HttpMethod,
        endpoint: str,
        *,
        json: Any = None,
        params: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
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
        with self._http_client(verify) as client:
            return client.request(method, url, **request_kwargs)

    def _make_multipart_request(
        self,
        method: HttpMethod,
        endpoint: str,
        *,
        zip_file_path: str | Path,
        data: dict[str, str] | None = None,
        timeout: float | None = None,
        verify: bool = True,
    ) -> httpx.Response:
        """Send an authenticated multipart request containing a project ZIP."""
        url = urljoin(self.base_url, endpoint)
        headers = dict(cast(dict[str, str], self.headers))
        headers.pop("Content-Type", None)
        path = Path(zip_file_path)
        request_kwargs: dict[str, Any] = {"headers": headers}
        if data is not None:
            request_kwargs["data"] = data
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        with (
            path.open("rb") as file_handle,
            self._http_client(verify) as client,
        ):
            files = {
                "zip_file": (path.name, file_handle, "application/zip"),
            }
            return client.request(method, url, files=files, **request_kwargs)

    def login_to_tool_repository(
        self, user_identifier: str | None = None, project_id: str | None = None
    ) -> httpx.Response:
        """Log in to the tool repository.

        This request is authenticated, so sending user_identifier and project_id
        alongside it links the account to the local pseudonymous user id and to
        the project the command was run from - letting prior anonymous usage of
        that project be attributed after signup.

        Args:
            user_identifier: Local pseudonymous user id.
            project_id: ``[tool.crewai].project_id`` of the current project.
        """
        payload: LoginPayload = {}
        if user_identifier:
            payload["user_identifier"] = user_identifier
        if project_id:
            payload["project_id"] = project_id
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}/login", json=payload)

    def get_tool(self, handle: str) -> httpx.Response:
        return self._make_request("GET", f"{self.TOOLS_RESOURCE}/{handle}")

    def get_agent(self, handle: str) -> httpx.Response:
        return self._make_request("GET", f"{self.AGENTS_RESOURCE}/{handle}")

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

    def get_skill(
        self, org: str, name: str, version: str | None = None
    ) -> httpx.Response:
        params: dict[str, str] = {}
        if version is not None:
            params["version"] = version
        return self._make_request(
            "GET",
            f"{self.SKILLS_RESOURCE}/{org}/{name}",
            params=params or None,
        )

    def publish_skill(
        self,
        org: str,
        name: str,
        version: str,
        is_public: bool,
        description: str | None,
        encoded_file: str,
    ) -> httpx.Response:
        payload = {
            "org": org,
            "name": name,
            "version": version,
            "public": is_public,
            "description": description,
            "file": encoded_file,
        }
        return self._make_request("POST", self.SKILLS_RESOURCE, json=payload)

    def list_skills(self, org: str | None = None) -> httpx.Response:
        params: dict[str, str] = {}
        if org is not None:
            params["org"] = org
        return self._make_request(
            "GET",
            self.SKILLS_RESOURCE,
            params=params or None,
        )

    def deploy_by_name(
        self, project_name: str, *, project_id: str | None = None, force: bool = False
    ) -> httpx.Response:
        """Redeploy by name, with PROJECT_ID (`[tool.crewai].project_id`) when set.

        The body is sent only when there is something in it. An AMP that
        predates project ids ignores it: this route has read its body through
        Rails' ``params.permit`` since the v1 API (2024-08), which drops a key
        it does not name, and AMP never made unpermitted keys an error. FORCE
        redeploys even while a models comparison runs on the deployment (AMP
        answers 409 ``comparison_running`` otherwise; an older AMP ignores it).
        """
        endpoint = f"{self.CREWS_RESOURCE}/by-name/{project_name}/deploy"
        return self._redeploy(endpoint, project_id=project_id, force=force)

    def deploy_by_uuid(
        self, uuid: str, *, project_id: str | None = None, force: bool = False
    ) -> httpx.Response:
        """Redeploy by uuid, with PROJECT_ID and FORCE as for `deploy_by_name`."""
        endpoint = f"{self.CREWS_RESOURCE}/{uuid}/deploy"
        return self._redeploy(endpoint, project_id=project_id, force=force)

    def _redeploy(
        self, endpoint: str, *, project_id: str | None, force: bool
    ) -> httpx.Response:
        body: dict[str, Any] = {}
        if project_id:
            body["project_id"] = project_id
        if force:
            body["force"] = True
        if body:
            return self._make_request("POST", endpoint, json=body)
        return self._make_request("POST", endpoint)

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

    def create_crew_from_zip(
        self,
        zip_file_path: str | Path,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        project_id: str | None = None,
    ) -> httpx.Response:
        """Create a crew deployment from a local project ZIP archive.

        PROJECT_ID is `[tool.crewai].project_id`, so AMP knows which project the
        deployment runs; sent only when the project has one.
        """
        data: dict[str, str] = {}
        if name:
            data["name"] = name
        if project_id:
            data["project_id"] = project_id
        if env:
            data.update({f"env[{key}]": value for key, value in env.items()})
        return self._make_multipart_request(
            "POST",
            f"{self.CREWS_RESOURCE}/zip",
            zip_file_path=zip_file_path,
            data=data or None,
            timeout=300,
        )

    def update_crew_from_zip(
        self,
        uuid: str,
        zip_file_path: str | Path,
        *,
        env: dict[str, str] | None = None,
        project_id: str | None = None,
        force: bool = False,
    ) -> httpx.Response:
        """Update an existing crew deployment from a local project ZIP archive.

        PROJECT_ID as for `create_crew_from_zip`, FORCE as for `deploy_by_name`.
        """
        data: dict[str, str] = {}
        if project_id:
            data["project_id"] = project_id
        if force:
            data["force"] = "true"
        if env:
            data.update({f"env[{key}]": value for key, value in env.items()})
        return self._make_multipart_request(
            "POST",
            f"{self.CREWS_RESOURCE}/{uuid}/zip_update",
            zip_file_path=zip_file_path,
            data=data or None,
            timeout=300,
        )

    def get_organizations(self) -> httpx.Response:
        return self._make_request("GET", self.ORGANIZATIONS_RESOURCE)

    def get_integration_token(self) -> str:
        """Return the Enterprise Action Auth Token for the active organization."""
        response = self._make_request(
            "GET", self.INTEGRATION_TOKEN_RESOURCE, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("api_key") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("AMP did not return an integration token")
        return token

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
        return self._make_request(
            "GET", f"{self.INTEGRATIONS_RESOURCE}/apps", timeout=_TRIGGER_TIMEOUT
        )

    def get_trigger_payload(self, app_slug: str, trigger_slug: str) -> httpx.Response:
        """Get sample payload for a specific trigger."""
        return self._make_request(
            "GET",
            f"{self.INTEGRATIONS_RESOURCE}/{app_slug}/{trigger_slug}/payload",
            timeout=_TRIGGER_TIMEOUT,
        )

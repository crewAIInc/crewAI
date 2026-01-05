from typing import Any
from urllib.parse import urljoin
import os
import requests

from crewai.cli.config import Settings
from crewai.cli.constants import DEFAULT_CREWAI_ENTERPRISE_URL
from crewai.cli.version import get_crewai_version


class PlusAPI:
    """
    This class exposes methods for working with the CrewAI+ API.
    """

    TOOLS_RESOURCE = "/crewai_plus/api/v1/tools"
    ORGANIZATIONS_RESOURCE = "/crewai_plus/api/v1/me/organizations"
    CREWS_RESOURCE = "/crewai_plus/api/v1/crews"
    AGENTS_RESOURCE = "/crewai_plus/api/v1/agents"
    TRACING_RESOURCE = "/crewai_plus/api/v1/tracing"
    EPHEMERAL_TRACING_RESOURCE = "/crewai_plus/api/v1/tracing/ephemeral"
    INTEGRATIONS_RESOURCE = "/crewai_plus/api/v1/integrations"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"CrewAI-CLI/{get_crewai_version()}",
            "X-Crewai-Version": get_crewai_version(),
        }
        settings = Settings()
        if settings.org_uuid:
            self.headers["X-Crewai-Organization-Id"] = settings.org_uuid

        self.base_url = os.getenv("CREWAI_PLUS_URL") or str(settings.enterprise_base_url) or DEFAULT_CREWAI_ENTERPRISE_URL

    def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> requests.Response:
        url = urljoin(self.base_url, endpoint)
        session = requests.Session()
        session.trust_env = False
        return session.request(method, url, headers=self.headers, **kwargs)

    def login_to_tool_repository(self) -> requests.Response:
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}/login")

    def get_tool(self, handle: str) -> requests.Response:
        return self._make_request("GET", f"{self.TOOLS_RESOURCE}/{handle}")

    def get_agent(self, handle: str) -> requests.Response:
        return self._make_request("GET", f"{self.AGENTS_RESOURCE}/{handle}")

    def publish_tool(
        self,
        handle: str,
        is_public: bool,
        version: str,
        description: str | None,
        encoded_file: str,
        available_exports: list[dict[str, Any]] | None = None,
    ) -> requests.Response:
        params = {
            "handle": handle,
            "public": is_public,
            "version": version,
            "file": encoded_file,
            "description": description,
            "available_exports": available_exports,
        }
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}", json=params)

    def deploy_by_name(self, project_name: str) -> requests.Response:
        return self._make_request(
            "POST", f"{self.CREWS_RESOURCE}/by-name/{project_name}/deploy"
        )

    def deploy_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("POST", f"{self.CREWS_RESOURCE}/{uuid}/deploy")

    def crew_status_by_name(self, project_name: str) -> requests.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/by-name/{project_name}/status"
        )

    def crew_status_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("GET", f"{self.CREWS_RESOURCE}/{uuid}/status")

    def crew_by_name(
        self, project_name: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/by-name/{project_name}/logs/{log_type}"
        )

    def crew_by_uuid(
        self, uuid: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request(
            "GET", f"{self.CREWS_RESOURCE}/{uuid}/logs/{log_type}"
        )

    def delete_crew_by_name(self, project_name: str) -> requests.Response:
        return self._make_request(
            "DELETE", f"{self.CREWS_RESOURCE}/by-name/{project_name}"
        )

    def delete_crew_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("DELETE", f"{self.CREWS_RESOURCE}/{uuid}")

    def list_crews(self) -> requests.Response:
        return self._make_request("GET", self.CREWS_RESOURCE)

    def create_crew(self, payload: dict[str, Any]) -> requests.Response:
        return self._make_request("POST", self.CREWS_RESOURCE, json=payload)

    def get_organizations(self) -> requests.Response:
        return self._make_request("GET", self.ORGANIZATIONS_RESOURCE)

    def initialize_trace_batch(self, payload: dict[str, Any]) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.TRACING_RESOURCE}/batches",
            json=payload,
            timeout=30,
        )

    def initialize_ephemeral_trace_batch(
        self, payload: dict[str, Any]
    ) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches",
            json=payload,
        )

    def send_trace_events(
        self, trace_batch_id: str, payload: dict[str, Any]
    ) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
            timeout=30,
        )

    def send_ephemeral_trace_events(
        self, trace_batch_id: str, payload: dict[str, Any]
    ) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
            timeout=30,
        )

    def finalize_trace_batch(
        self, trace_batch_id: str, payload: dict[str, Any]
    ) -> requests.Response:
        return self._make_request(
            "PATCH",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
            timeout=30,
        )

    def finalize_ephemeral_trace_batch(
        self, trace_batch_id: str, payload: dict[str, Any]
    ) -> requests.Response:
        return self._make_request(
            "PATCH",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
            timeout=30,
        )

    def mark_trace_batch_as_failed(
        self, trace_batch_id: str, error_message: str
    ) -> requests.Response:
        return self._make_request(
            "PATCH",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}",
            json={"status": "failed", "failure_reason": error_message},
            timeout=30,
        )

    def get_triggers(self) -> requests.Response:
        """Get all available triggers from integrations."""
        return self._make_request("GET", f"{self.INTEGRATIONS_RESOURCE}/apps")

    def get_trigger_payload(
        self, app_slug: str, trigger_slug: str
    ) -> requests.Response:
        """Get sample payload for a specific trigger."""
        return self._make_request(
            "GET", f"{self.INTEGRATIONS_RESOURCE}/{app_slug}/{trigger_slug}/payload"
        )

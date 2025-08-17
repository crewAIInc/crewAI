from typing import List, Optional
from urllib.parse import urljoin

import requests

from crewai.cli.config import Settings
from crewai.cli.version import get_crewai_version
from crewai.cli.constants import DEFAULT_CREWAI_ENTERPRISE_URL


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

        self.base_url = (
            str(settings.enterprise_base_url) or DEFAULT_CREWAI_ENTERPRISE_URL
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = urljoin(self.base_url, endpoint)
        session = requests.Session()
        session.trust_env = False
        return session.request(method, url, headers=self.headers, **kwargs)

    def login_to_tool_repository(self):
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}/login")

    def get_tool(self, handle: str):
        return self._make_request("GET", f"{self.TOOLS_RESOURCE}/{handle}")

    def get_agent(self, handle: str):
        return self._make_request("GET", f"{self.AGENTS_RESOURCE}/{handle}")

    def publish_tool(
        self,
        handle: str,
        is_public: bool,
        version: str,
        description: Optional[str],
        encoded_file: str,
        available_exports: Optional[List[str]] = None,
    ):
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

    def create_crew(self, payload) -> requests.Response:
        return self._make_request("POST", self.CREWS_RESOURCE, json=payload)

    def get_organizations(self) -> requests.Response:
        return self._make_request("GET", self.ORGANIZATIONS_RESOURCE)

    def send_trace_batch(self, payload) -> requests.Response:
        return self._make_request("POST", self.TRACING_RESOURCE, json=payload)

    def initialize_trace_batch(self, payload) -> requests.Response:
        return self._make_request(
            "POST", f"{self.TRACING_RESOURCE}/batches", json=payload
        )

    def initialize_ephemeral_trace_batch(self, payload) -> requests.Response:
        return self._make_request(
            "POST", f"{self.EPHEMERAL_TRACING_RESOURCE}/batches", json=payload
        )

    def send_trace_events(self, trace_batch_id: str, payload) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
        )

    def send_ephemeral_trace_events(
        self, trace_batch_id: str, payload
    ) -> requests.Response:
        return self._make_request(
            "POST",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/events",
            json=payload,
        )

    def finalize_trace_batch(self, trace_batch_id: str, payload) -> requests.Response:
        return self._make_request(
            "PATCH",
            f"{self.TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
        )

    def finalize_ephemeral_trace_batch(
        self, trace_batch_id: str, payload
    ) -> requests.Response:
        return self._make_request(
            "PATCH",
            f"{self.EPHEMERAL_TRACING_RESOURCE}/batches/{trace_batch_id}/finalize",
            json=payload,
        )

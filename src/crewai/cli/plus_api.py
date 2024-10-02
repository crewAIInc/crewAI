from typing import Optional
import requests
from os import getenv
from crewai.cli.utils import get_crewai_version
from urllib.parse import urljoin


class PlusAPI:
    """
    This class exposes methods for working with the CrewAI+ API.
    """

    TOOLS_RESOURCE = "/crewai_plus/api/v1/tools"
    CREWS_RESOURCE = "/crewai_plus/api/v1/crews"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"CrewAI-CLI/{get_crewai_version()}",
            "X-Crewai-Version": get_crewai_version(),
        }
        self.base_url = getenv("CREWAI_BASE_URL", "https://app.crewai.com")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = urljoin(self.base_url, endpoint)
        return requests.request(method, url, headers=self.headers, **kwargs)

    def login_to_tool_repository(self):
        return self._make_request("POST", f"{self.TOOLS_RESOURCE}/login")

    def get_tool(self, handle: str):
        return self._make_request("GET", f"{self.TOOLS_RESOURCE}/{handle}")

    def publish_tool(
        self,
        handle: str,
        is_public: bool,
        version: str,
        description: Optional[str],
        encoded_file: str,
    ):
        params = {
            "handle": handle,
            "public": is_public,
            "version": version,
            "file": encoded_file,
            "description": description,
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

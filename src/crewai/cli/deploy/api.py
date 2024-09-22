from os import getenv

import requests

from crewai.cli.deploy.utils import get_crewai_version


class CrewAPI:
    """
    CrewAPI class to interact with the crewAI+ API.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"CrewAI-CLI/{get_crewai_version()}",
        }
        self.base_url = getenv(
            "CREWAI_BASE_URL", "https://crewai.com/crewai_plus/api/v1/crews"
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        return requests.request(method, url, headers=self.headers, **kwargs)

    # Deploy
    def deploy_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("POST", f"by-name/{project_name}/deploy")

    def deploy_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("POST", f"{uuid}/deploy")

    # Status
    def status_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("GET", f"by-name/{project_name}/status")

    def status_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("GET", f"{uuid}/status")

    # Logs
    def logs_by_name(
        self, project_name: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request("GET", f"by-name/{project_name}/logs/{log_type}")

    def logs_by_uuid(
        self, uuid: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request("GET", f"{uuid}/logs/{log_type}")

    # Delete
    def delete_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("DELETE", f"by-name/{project_name}")

    def delete_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("DELETE", f"{uuid}")

    # List
    def list_crews(self) -> requests.Response:
        return self._make_request("GET", "")

    # Create
    def create_crew(self, payload) -> requests.Response:
        return self._make_request("POST", "", json=payload)

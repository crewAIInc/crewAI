from os import getenv

import requests


class CrewAPI:
    """
    CrewAPI class to interact with the crewAI+ API.
    """

    CREW_BASE_URL = getenv("BASE_URL", "http://localhost:3000/crewai_plus/api/v1/crews")
    MAIN_BASE_URL = getenv("MAIN_BASE_URL", "http://localhost:3000/crewai_plus/api/v1")

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self, method: str, endpoint: str, base_url: str = CREW_BASE_URL, **kwargs
    ) -> requests.Response:
        url = f"{base_url}/{endpoint}"
        return requests.request(method, url, headers=self.headers, **kwargs)

    def deploy_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("POST", f"by-name/{project_name}/deploy")

    def deploy_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("POST", f"{uuid}/deploy")

    def status_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("GET", f"by-name/{project_name}/status")

    def status_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("GET", f"{uuid}/status")

    def logs_by_name(
        self, project_name: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request("GET", f"by-name/{project_name}/logs/{log_type}")

    def logs_by_uuid(
        self, uuid: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request("GET", f"{uuid}/logs/{log_type}")

    def delete_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("DELETE", f"by-name/{project_name}")

    def delete_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("DELETE", f"{uuid}")

    def list_crews(self) -> requests.Response:
        return self._make_request("GET", "")

    def create_crew(self, payload) -> requests.Response:
        return self._make_request("POST", "", json=payload)

    def signup(self) -> requests.Response:
        return self._make_request("GET", "signup_link", base_url=self.MAIN_BASE_URL)

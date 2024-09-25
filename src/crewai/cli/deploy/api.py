import requests
from crewai.cli.plus_api import PlusAPI


class CrewAPI(PlusAPI):
    """
    CrewAPI class to interact with the Crew resource in CrewAI+ API.
    """

    RESOURCE = "/crewai_plus/api/v1/crews"

    # Deploy
    def deploy_by_name(self, project_name: str) -> requests.Response:
        return self._make_request(
            "POST", f"{self.RESOURCE}/by-name/{project_name}/deploy"
        )

    def deploy_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("POST", f"{self.RESOURCE}/{uuid}/deploy")

    # Status
    def status_by_name(self, project_name: str) -> requests.Response:
        return self._make_request(
            "GET", f"{self.RESOURCE}/by-name/{project_name}/status"
        )

    def status_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("GET", f"{self.RESOURCE}/{uuid}/status")

    # Logs
    def logs_by_name(
        self, project_name: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request(
            "GET", f"{self.RESOURCE}/by-name/{project_name}/logs/{log_type}"
        )

    def logs_by_uuid(
        self, uuid: str, log_type: str = "deployment"
    ) -> requests.Response:
        return self._make_request("GET", f"{self.RESOURCE}/{uuid}/logs/{log_type}")

    # Delete
    def delete_by_name(self, project_name: str) -> requests.Response:
        return self._make_request("DELETE", f"{self.RESOURCE}/by-name/{project_name}")

    def delete_by_uuid(self, uuid: str) -> requests.Response:
        return self._make_request("DELETE", f"{self.RESOURCE}/{uuid}")

    # List
    def list_crews(self) -> requests.Response:
        return self._make_request("GET", self.RESOURCE)

    # Create
    def create_crew(self, payload) -> requests.Response:
        return self._make_request("POST", self.RESOURCE, json=payload)

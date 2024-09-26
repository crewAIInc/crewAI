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

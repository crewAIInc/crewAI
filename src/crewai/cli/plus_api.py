import requests
from os import getenv
from crewai.cli.deploy.utils import get_crewai_version
from urllib.parse import urljoin

class PlusAPI:
    """
    This class exposes methods for working with the CrewAI+ API.
    """

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

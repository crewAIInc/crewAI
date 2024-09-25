from typing import Optional
from crewai.cli.plus_api import PlusAPI


class ToolsAPI(PlusAPI):
    RESOURCE = "/crewai_plus/api/v1/tools"

    def get(self, handle: str):
        return self._make_request("GET", f"{self.RESOURCE}/{handle}")

    def publish(
        self,
        handle: str,
        public: bool,
        version: str,
        description: Optional[str],
        encoded_file: str,
    ):
        params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
        }
        return self._make_request("POST", f"{self.RESOURCE}", json=params)

"""CrewAI CLI API client extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urljoin

from crewai_core.plus_api import PlusAPI as _CorePlusAPI
import httpx


HttpMethod = Literal["GET", "POST", "PATCH", "DELETE"]


class PlusAPI(_CorePlusAPI):
    """CLI API client.

    The ZIP deployment methods live here as well as in newer crewai-core
    versions so editable CLI installs still work when an older crewai-core is
    present in the runtime environment.
    """

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
            httpx.Client(trust_env=False, verify=verify) as client,
        ):
            files = {
                "zip_file": (path.name, file_handle, "application/zip"),
            }
            return client.request(method, url, files=files, **request_kwargs)

    def create_crew_from_zip(
        self,
        zip_file_path: str | Path,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Create a crew deployment from a local project ZIP archive."""
        data: dict[str, str] = {}
        if name:
            data["name"] = name
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
    ) -> httpx.Response:
        """Update an existing crew deployment from a local project ZIP archive."""
        data: dict[str, str] = {}
        if env:
            data.update({f"env[{key}]": value for key, value in env.items()})
        return self._make_multipart_request(
            "POST",
            f"{self.CREWS_RESOURCE}/{uuid}/zip_update",
            zip_file_path=zip_file_path,
            data=data or None,
            timeout=300,
        )


__all__ = ["PlusAPI"]

"""Crewai Enterprise Tools."""

import json
import os
import re
import tempfile
from typing import Any

from crewai.tools import BaseTool
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from pydantic import Field, create_model
import requests

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)

_FILE_MARKER_PREFIX = "__CREWAI_FILE__"

_MIME_TO_EXTENSION = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/json": ".json",
    "application/zip": ".zip",
}


class CrewAIPlatformActionTool(BaseTool):
    action_name: str = Field(default="", description="The name of the action")
    action_schema: dict[str, Any] = Field(
        default_factory=dict, description="The schema of the action"
    )

    def __init__(
        self,
        description: str,
        action_name: str,
        action_schema: dict[str, Any],
    ):
        parameters = action_schema.get("function", {}).get("parameters", {})

        if parameters and parameters.get("properties"):
            try:
                if "title" not in parameters:
                    parameters = {**parameters, "title": f"{action_name}Schema"}
                if "type" not in parameters:
                    parameters = {**parameters, "type": "object"}
                args_schema = create_model_from_schema(parameters)
            except Exception:
                args_schema = create_model(f"{action_name}Schema")
        else:
            args_schema = create_model(f"{action_name}Schema")

        super().__init__(
            name=action_name.lower().replace(" ", "_"),
            description=description,
            args_schema=args_schema,
        )
        self.action_name = action_name
        self.action_schema = action_schema

    def _run(self, **kwargs: Any) -> str:
        try:
            cleaned_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            api_url = (
                f"{get_platform_api_base_url()}/actions/{self.action_name}/execute"
            )
            token = get_platform_integration_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {
                "integration": cleaned_kwargs if cleaned_kwargs else {"_noop": True}
            }

            response = requests.post(
                url=api_url,
                headers=headers,
                json=payload,
                timeout=300,
                stream=True,
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )

            content_type = response.headers.get("Content-Type", "")

            # Check if response is binary (non-JSON)
            if "application/json" not in content_type:
                return self._handle_binary_response(response)

            # Normal JSON response
            data = response.json()
            if not response.ok:
                if isinstance(data, dict):
                    error_info = data.get("error", {})
                    if isinstance(error_info, dict):
                        error_message = error_info.get("message", json.dumps(data))
                    else:
                        error_message = str(error_info)
                else:
                    error_message = str(data)
                return f"API request failed: {error_message}"

            return json.dumps(data, indent=2)

        except Exception as e:
            return f"Error executing action {self.action_name}: {e!s}"

    def _handle_binary_response(self, response: requests.Response) -> str:
        """Handle binary streaming response from the API.

        Streams the binary content to a temporary file and returns a marker
        that can be processed by the file hook to inject the file into the
        LLM context.

        Args:
            response: The streaming HTTP response with binary content.

        Returns:
            A file marker string in the format:
            __CREWAI_FILE__:filename:content_type:file_path
        """
        content_type = response.headers.get("Content-Type", "application/octet-stream")

        filename = self._extract_filename_from_headers(response.headers)

        extension = self._get_file_extension(content_type, filename)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=extension, prefix="crewai_"
        ) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        return f"{_FILE_MARKER_PREFIX}:{filename}:{content_type}:{tmp_path}"

    def _extract_filename_from_headers(
        self, headers: requests.structures.CaseInsensitiveDict
    ) -> str:
        content_disposition = headers.get("Content-Disposition", "")
        if content_disposition:
            match = re.search(r'filename="?([^";\s]+)"?', content_disposition)
            if match:
                return match.group(1)
        return "downloaded_file"

    def _get_file_extension(self, content_type: str, filename: str) -> str:
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1]

        base_content_type = content_type.split(";")[0].strip()
        return _MIME_TO_EXTENSION.get(base_content_type, "")

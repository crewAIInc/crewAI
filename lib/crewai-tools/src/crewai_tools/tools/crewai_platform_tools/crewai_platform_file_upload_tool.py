"""CrewAI Platform File Upload Tool.

Uploads a file from disk to Google Drive without passing file content
through the LLM context window. This avoids token waste and context
limit issues with large or binary files.
"""

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import Field, create_model

from crewai_tools.tools.crewai_platform_tools.misc import (
    get_platform_api_base_url,
    get_platform_integration_token,
)

import requests


logger = logging.getLogger(__name__)

# Google Drive simple upload limit
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


_UploadFromFileSchema = create_model(
    "UploadFromFileSchema",
    file_path=(str, Field(description="Path to the local file to upload")),
    name=(
        Optional[str],
        Field(
            default=None,
            description="Name for the file in Google Drive (defaults to the local filename)",
        ),
    ),
    mime_type=(
        Optional[str],
        Field(
            default=None,
            description="MIME type of the file (auto-detected if not provided)",
        ),
    ),
    parent_folder_id=(
        Optional[str],
        Field(
            default=None,
            description="ID of the parent folder where the file should be created",
        ),
    ),
    description=(
        Optional[str],
        Field(default=None, description="Description of the file"),
    ),
)


class CrewAIPlatformFileUploadTool(BaseTool):
    """Upload a file from disk to Google Drive.

    Reads the file locally and sends it to the platform API, bypassing
    the LLM context window entirely. Supports auto-detection of MIME type
    and optional file naming.
    """

    name: str = "google_drive_upload_from_file"
    description: str = (
        "Upload a file from a local path to Google Drive. "
        "The file is read directly from disk — its content never passes "
        "through the LLM context, making this efficient for large or binary files."
    )
    args_schema: type = _UploadFromFileSchema

    def _run(
        self,
        file_path: str,
        name: str | None = None,
        mime_type: str | None = None,
        parent_folder_id: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            path = Path(file_path).expanduser().resolve()

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not path.is_file():
                return f"Error: Path is not a file: {file_path}"

            file_size = path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                return (
                    f"Error: File size ({file_size / (1024 * 1024):.1f} MB) exceeds "
                    f"the 50 MB limit for simple uploads. Consider splitting the file "
                    f"or using a resumable upload method."
                )

            # Read and encode file content
            content_bytes = path.read_bytes()
            content_b64 = base64.b64encode(content_bytes).decode("utf-8")

            # Auto-detect MIME type if not provided
            if mime_type is None:
                guessed_type, _ = mimetypes.guess_type(str(path))
                mime_type = guessed_type or "application/octet-stream"

            # Use filename if name not provided
            upload_name = name or path.name

            # Build payload matching the existing upload_file action format
            payload_data: dict[str, Any] = {
                "name": upload_name,
                "content": content_b64,
            }
            if mime_type:
                payload_data["mime_type"] = mime_type
            if parent_folder_id:
                payload_data["parent_folder_id"] = parent_folder_id
            if description:
                payload_data["description"] = description

            api_url = (
                f"{get_platform_api_base_url()}/actions/GOOGLE_DRIVE_SAVE_FILE/execute"
            )
            token = get_platform_integration_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            payload = {"integration": payload_data}

            response = requests.post(
                url=api_url,
                headers=headers,
                json=payload,
                timeout=120,  # Longer timeout for file uploads
                verify=os.environ.get("CREWAI_FACTORY", "false").lower() != "true",
            )

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

        except PermissionError:
            return f"Error: Permission denied reading file: {file_path}"
        except Exception as e:
            return f"Error uploading file: {e!s}"

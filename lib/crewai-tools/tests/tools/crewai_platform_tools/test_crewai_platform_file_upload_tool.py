"""Tests for CrewAIPlatformFileUploadTool."""

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool import (
    CrewAIPlatformFileUploadTool,
    MAX_FILE_SIZE_BYTES,
)


class TestCrewAIPlatformFileUploadTool:
    """Test suite for the file upload tool."""

    def setup_method(self):
        self.tool = CrewAIPlatformFileUploadTool()

    def test_tool_name_and_description(self):
        assert self.tool.name == "google_drive_upload_from_file"
        assert "local path" in self.tool.description.lower()

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_successful_upload(self, mock_post):
        """Test uploading a file successfully."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "file123",
            "name": "test.txt",
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, Google Drive!")
            temp_path = f.name

        try:
            result = self.tool._run(file_path=temp_path)

            assert "file123" in result
            mock_post.assert_called_once()

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]["integration"]
            assert payload["name"] == Path(temp_path).name
            assert payload["mime_type"] == "text/plain"
            # Verify content is base64-encoded
            decoded = base64.b64decode(payload["content"]).decode("utf-8")
            assert decoded == "Hello, Google Drive!"
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_custom_name_override(self, mock_post):
        """Test that a custom name overrides the filename."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path, name="custom_report.txt")

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert payload["name"] == "custom_report.txt"
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_mime_type_auto_detection(self, mock_post):
        """Test MIME type is auto-detected from file extension."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{}")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path)

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert payload["mime_type"] == "application/json"
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_custom_mime_type(self, mock_post):
        """Test that a custom MIME type overrides auto-detection."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path, mime_type="application/pdf")

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert payload["mime_type"] == "application/pdf"
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test error when file does not exist."""
        result = self.tool._run(file_path="/nonexistent/path/file.txt")
        assert "Error: File not found" in result

    def test_path_is_directory(self):
        """Test error when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.tool._run(file_path=tmpdir)
            assert "Error: Path is not a file" in result

    def test_file_too_large(self):
        """Test error when file exceeds 50 MB limit."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            # Create a sparse file that reports as > 50 MB
            f.seek(MAX_FILE_SIZE_BYTES + 1)
            f.write(b"\0")

        try:
            result = self.tool._run(file_path=temp_path)
            assert "exceeds" in result
            assert "50 MB" in result
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_optional_params_included(self, mock_post):
        """Test that optional params are included in payload when provided."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(
                file_path=temp_path,
                parent_folder_id="folder456",
                description="My report",
            )

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert payload["parent_folder_id"] == "folder456"
            assert payload["description"] == "My report"
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_optional_params_excluded_when_none(self, mock_post):
        """Test that optional params are NOT in payload when not provided."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path)

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert "parent_folder_id" not in payload
            assert "description" not in payload
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_api_error_response(self, mock_post):
        """Test handling of API error responses."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.json.return_value = {
            "error": {"message": "Unauthorized"}
        }
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            result = self.tool._run(file_path=temp_path)
            assert "API request failed" in result
            assert "Unauthorized" in result
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
            "CREWAI_FACTORY": "true",
        },
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_ssl_verification_disabled_for_factory(self, mock_post):
        """Test that SSL verification is disabled when CREWAI_FACTORY=true."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path)
            assert mock_post.call_args.kwargs["verify"] is False
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_ssl_verification_enabled_by_default(self, mock_post):
        """Test that SSL verification is enabled by default."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("content")
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path)
            assert mock_post.call_args.kwargs["verify"] is True
        finally:
            os.unlink(temp_path)

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_file_upload_tool.requests.post"
    )
    def test_binary_file_upload(self, mock_post):
        """Test uploading a binary file (e.g. PNG)."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"id": "file123"}
        mock_post.return_value = mock_response

        binary_content = bytes(range(256))

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as f:
            f.write(binary_content)
            temp_path = f.name

        try:
            self.tool._run(file_path=temp_path)

            payload = mock_post.call_args.kwargs["json"]["integration"]
            assert payload["mime_type"] == "image/png"
            decoded = base64.b64decode(payload["content"])
            assert decoded == binary_content
        finally:
            os.unlink(temp_path)

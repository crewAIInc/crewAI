"""Tests for AddImageTool functionality."""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from crewai.tools.agent_tools.add_image_tool import AddImageTool


class TestAddImageToolNormalizeImageUrl:
    """Tests for the _normalize_image_url method."""

    def test_http_url_unchanged(self):
        """HTTP URLs should be returned unchanged."""
        tool = AddImageTool()
        url = "http://example.com/image.png"
        result = tool._normalize_image_url(url)
        assert result == url

    def test_https_url_unchanged(self):
        """HTTPS URLs should be returned unchanged."""
        tool = AddImageTool()
        url = "https://example.com/image.jpg"
        result = tool._normalize_image_url(url)
        assert result == url

    def test_data_url_unchanged(self):
        """Data URLs should be returned unchanged."""
        tool = AddImageTool()
        url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result = tool._normalize_image_url(url)
        assert result == url

    def test_local_file_converted_to_base64(self):
        """Local file paths should be converted to base64 data URLs."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            f.write(test_image_data)
            temp_path = f.name

        try:
            result = tool._normalize_image_url(temp_path)

            assert result.startswith("data:image/png;base64,")
            encoded_data = result.split(",")[1]
            decoded_data = base64.b64decode(encoded_data)
            assert decoded_data == test_image_data
        finally:
            os.unlink(temp_path)

    def test_local_file_with_jpeg_extension(self):
        """JPEG files should have correct mime type."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
            f.write(test_image_data)
            temp_path = f.name

        try:
            result = tool._normalize_image_url(temp_path)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(temp_path)

    def test_local_file_with_gif_extension(self):
        """GIF files should have correct mime type."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            test_image_data = b"GIF89a"
            f.write(test_image_data)
            temp_path = f.name

        try:
            result = tool._normalize_image_url(temp_path)
            assert result.startswith("data:image/gif;base64,")
        finally:
            os.unlink(temp_path)

    def test_local_file_with_webp_extension(self):
        """WebP files should be converted to base64 data URLs."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            test_image_data = b"RIFF\x00\x00\x00\x00WEBP"
            f.write(test_image_data)
            temp_path = f.name

        try:
            result = tool._normalize_image_url(temp_path)
            assert result.startswith("data:")
            assert ";base64," in result
        finally:
            os.unlink(temp_path)

    def test_file_url_converted_to_base64(self):
        """file:// URLs should be converted to base64 data URLs."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            f.write(test_image_data)
            temp_path = f.name

        try:
            file_url = f"file://{temp_path}"
            result = tool._normalize_image_url(file_url)

            assert result.startswith("data:image/png;base64,")
            encoded_data = result.split(",")[1]
            decoded_data = base64.b64decode(encoded_data)
            assert decoded_data == test_image_data
        finally:
            os.unlink(temp_path)

    def test_relative_path_converted_to_base64(self):
        """Relative file paths should be converted to base64 data URLs."""
        tool = AddImageTool()

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                relative_path = "test_image.png"
                with open(relative_path, "wb") as f:
                    f.write(test_image_data)

                result = tool._normalize_image_url(relative_path)

                assert result.startswith("data:image/png;base64,")
                encoded_data = result.split(",")[1]
                decoded_data = base64.b64decode(encoded_data)
                assert decoded_data == test_image_data
            finally:
                os.chdir(original_cwd)

    def test_tilde_path_expanded(self):
        """Paths with ~ should be expanded to home directory."""
        tool = AddImageTool()

        home_dir = Path.home()
        with tempfile.NamedTemporaryFile(
            suffix=".png", dir=home_dir, delete=False
        ) as f:
            test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            f.write(test_image_data)
            temp_path = f.name
            filename = os.path.basename(temp_path)

        try:
            tilde_path = f"~/{filename}"
            result = tool._normalize_image_url(tilde_path)

            assert result.startswith("data:image/png;base64,")
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file_raises_error(self):
        """Non-existent file paths should raise FileNotFoundError."""
        tool = AddImageTool()

        with pytest.raises(FileNotFoundError) as exc_info:
            tool._normalize_image_url("/nonexistent/path/to/image.png")

        assert "Image file not found" in str(exc_info.value)
        assert "/nonexistent/path/to/image.png" in str(exc_info.value)

    def test_nonexistent_file_url_raises_error(self):
        """Non-existent file:// URLs should raise FileNotFoundError."""
        tool = AddImageTool()

        with pytest.raises(FileNotFoundError) as exc_info:
            tool._normalize_image_url("file:///nonexistent/path/to/image.png")

        assert "Image file not found" in str(exc_info.value)

    def test_unknown_extension_uses_mimetypes_or_defaults_to_png(self):
        """Files should use mimetypes guess or default to image/png."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".unknownext123", delete=False) as f:
            test_image_data = b"some binary data"
            f.write(test_image_data)
            temp_path = f.name

        try:
            result = tool._normalize_image_url(temp_path)
            assert result.startswith("data:")
            assert ";base64," in result
            encoded_data = result.split(",")[1]
            decoded_data = base64.b64decode(encoded_data)
            assert decoded_data == test_image_data
        finally:
            os.unlink(temp_path)

    def test_unknown_scheme_url_returned_unchanged(self):
        """URLs with unknown schemes (like s3://) should be returned unchanged."""
        tool = AddImageTool()
        url = "s3://bucket/path/to/image.png"
        result = tool._normalize_image_url(url)
        assert result == url

    def test_file_read_error_raises_value_error(self):
        """File read errors should raise ValueError with helpful message."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "is_file", return_value=True):
                        with pytest.raises(ValueError) as exc_info:
                            tool._normalize_image_url(temp_path)

                        assert "Failed to read image file" in str(exc_info.value)
                        assert "Permission denied" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestAddImageToolRun:
    """Tests for the _run method."""

    def test_run_with_http_url(self):
        """_run should work correctly with HTTP URLs."""
        tool = AddImageTool()
        url = "https://example.com/test-image.jpg"
        action = "Please analyze this image"

        result = tool._run(image_url=url, action=action)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == action
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == url

    def test_run_with_local_file(self):
        """_run should convert local files to base64 data URLs."""
        tool = AddImageTool()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            f.write(test_image_data)
            temp_path = f.name

        try:
            action = "Describe this image"
            result = tool._run(image_url=temp_path, action=action)

            assert result["role"] == "user"
            assert len(result["content"]) == 2
            assert result["content"][0]["type"] == "text"
            assert result["content"][0]["text"] == action
            assert result["content"][1]["type"] == "image_url"
            assert result["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        finally:
            os.unlink(temp_path)

    def test_run_with_default_action(self):
        """_run should use default action when none is provided."""
        tool = AddImageTool()
        url = "https://example.com/test-image.jpg"

        result = tool._run(image_url=url)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] is not None
        assert result["content"][1]["type"] == "image_url"

    def test_run_with_data_url(self):
        """_run should work correctly with data URLs."""
        tool = AddImageTool()
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        action = "What is in this image?"

        result = tool._run(image_url=data_url, action=action)

        assert result["role"] == "user"
        assert result["content"][1]["image_url"]["url"] == data_url

    def test_run_raises_error_for_nonexistent_file(self):
        """_run should raise FileNotFoundError for non-existent files."""
        tool = AddImageTool()

        with pytest.raises(FileNotFoundError):
            tool._run(image_url="/nonexistent/path/to/image.png")


class TestAddImageToolIntegration:
    """Integration tests for AddImageTool."""

    def test_tool_has_correct_schema(self):
        """Tool should have correct args schema."""
        tool = AddImageTool()

        schema = tool.args_schema.model_json_schema()
        assert "image_url" in schema["properties"]
        assert "action" in schema["properties"]
        assert schema["required"] == ["image_url"]

    def test_tool_run_method_works(self):
        """Tool's public run method should work correctly."""
        tool = AddImageTool()
        url = "https://example.com/test-image.jpg"

        result = tool.run(image_url=url, action="Test action")

        assert result["role"] == "user"
        assert len(result["content"]) == 2

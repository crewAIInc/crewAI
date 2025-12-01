"""Tests for the Image class."""

import base64
from pathlib import Path
import tempfile

import pytest

from crewai.multimodal.image import Image


def test_image_from_url():
    """Test creating Image from URL."""
    img = Image.from_url("https://example.com/image.png")
    assert img.source == "https://example.com/image.png"
    assert img.source_type == "url"
    assert img.media_type == "image/png"


def test_image_from_file(tmp_path):
    """Test creating Image from file path."""
    # Create a temporary image file
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake png data")
    
    img = Image.from_file(str(img_file))
    assert img.source_type == "file"
    assert img.media_type == "image/png"


def test_image_from_base64():
    """Test creating Image from base64 string."""
    b64_data = base64.b64encode(b"fake image data").decode("utf-8")
    img = Image.from_base64(b64_data, media_type="image/jpeg")
    
    assert img.source == b64_data
    assert img.source_type == "base64"
    assert img.media_type == "image/jpeg"


def test_image_from_binary():
    """Test creating Image from binary data."""
    binary_data = b"fake binary image data"
    img = Image.from_binary(binary_data)
    
    assert img.source == binary_data
    assert img.source_type == "binary"
    assert img.media_type == "image/png"


def test_image_from_placeholder():
    """Test creating Image placeholder."""
    img = Image.from_placeholder("user_image")
    
    assert img.placeholder == "user_image"
    assert img.media_type == "image/png"


def test_image_auto_infer_source_type_url():
    """Test automatic source type inference for URLs."""
    img = Image(source="https://example.com/img.png")
    assert img.source_type == "url"


def test_image_auto_infer_source_type_file():
    """Test automatic source type inference for file paths."""
    img = Image(source="/path/to/image.png")
    assert img.source_type == "file"


def test_image_auto_infer_source_type_data_url():
    """Test automatic source type inference for data URLs."""
    img = Image(source="data:image/png;base64,iVBORw0KGgo=")
    assert img.source_type == "data_url"


def test_image_auto_infer_source_type_base64():
    """Test automatic source type inference for base64."""
    long_base64 = "A" * 200  # Long string without path separators
    img = Image(source=long_base64)
    assert img.source_type == "base64"


def test_image_auto_infer_source_type_binary():
    """Test automatic source type inference for binary."""
    img = Image(source=b"binary data")
    assert img.source_type == "binary"


def test_to_data_url_from_url():
    """Test converting URL to data URL (returns as-is)."""
    img = Image.from_url("https://example.com/image.png")
    data_url = img.to_data_url()
    assert data_url == "https://example.com/image.png"


def test_to_data_url_from_base64():
    """Test converting base64 to data URL."""
    b64_data = base64.b64encode(b"test data").decode("utf-8")
    img = Image.from_base64(b64_data, media_type="image/jpeg")
    data_url = img.to_data_url()
    
    assert data_url.startswith("data:image/jpeg;base64,")
    assert b64_data in data_url


def test_to_data_url_from_binary():
    """Test converting binary to data URL."""
    binary_data = b"test binary data"
    img = Image.from_binary(binary_data, media_type="image/png")
    data_url = img.to_data_url()
    
    expected_b64 = base64.b64encode(binary_data).decode("utf-8")
    assert data_url == f"data:image/png;base64,{expected_b64}"


def test_to_data_url_from_file(tmp_path):
    """Test converting file to data URL."""
    # Create a temporary image file
    img_file = tmp_path / "test.png"
    test_data = b"fake png file content"
    img_file.write_bytes(test_data)
    
    img = Image.from_file(str(img_file))
    data_url = img.to_data_url()
    
    expected_b64 = base64.b64encode(test_data).decode("utf-8")
    assert data_url == f"data:image/png;base64,{expected_b64}"


def test_to_data_url_placeholder_raises():
    """Test that placeholder raises error when converting to data URL."""
    img = Image.from_placeholder("user_image")
    
    with pytest.raises(ValueError, match="Cannot convert placeholder"):
        img.to_data_url()


def test_to_data_url_none_source_raises():
    """Test that None source raises error."""
    img = Image(source=None)
    
    with pytest.raises(ValueError, match="Image source is None"):
        img.to_data_url()


def test_to_data_url_file_not_found():
    """Test that missing file raises FileNotFoundError."""
    img = Image.from_file("/nonexistent/path/image.png")
    
    with pytest.raises(FileNotFoundError):
        img.to_data_url()


def test_to_message_content():
    """Test converting image to message content format."""
    img = Image.from_url("https://example.com/image.png")
    content = img.to_message_content()
    
    assert content["type"] == "image_url"
    assert "image_url" in content
    assert content["image_url"]["url"] == "https://example.com/image.png"


def test_str_representation():
    """Test string representation of Image."""
    img = Image.from_url("https://example.com/very/long/path/to/image.png")
    str_repr = str(img)
    assert "url:" in str_repr
    
    img_placeholder = Image.from_placeholder("test_image")
    str_repr = str(img_placeholder)
    assert "placeholder=test_image" in str_repr

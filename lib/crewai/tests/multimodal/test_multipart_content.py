"""Tests for the MultipartContent class."""

import pytest

from crewai.multimodal.image import Image
from crewai.multimodal.multipart_content import MultipartContent


def test_multipart_content_empty():
    """Test creating empty MultipartContent."""
    content = MultipartContent()
    assert len(content) == 0
    assert not content.has_images()


def test_add_text():
    """Test adding text to MultipartContent."""
    content = MultipartContent()
    content.add_text("Hello world")
    content.add_text("Another line")
    
    assert len(content) == 2
    assert content.parts[0] == "Hello world"
    assert content.parts[1] == "Another line"


def test_add_image():
    """Test adding image to MultipartContent."""
    content = MultipartContent()
    img = Image.from_url("https://example.com/image.png")
    content.add_image(img)
    
    assert len(content) == 1
    assert content.has_images()
    assert isinstance(content.parts[0], Image)


def test_mixed_content():
    """Test mixing text and images."""
    content = MultipartContent()
    content.add_text("Check out this image:")
    content.add_image(Image.from_url("https://example.com/img.png"))
    content.add_text("Pretty cool, right?")
    
    assert len(content) == 3
    assert content.has_images()
    assert isinstance(content.parts[0], str)
    assert isinstance(content.parts[1], Image)
    assert isinstance(content.parts[2], str)


def test_to_message_content_text_only():
    """Test converting text-only content to message format."""
    content = MultipartContent()
    content.add_text("Hello")
    content.add_text("World")
    
    message_parts = content.to_message_content()
    
    assert len(message_parts) == 2
    assert message_parts[0] == {"type": "text", "text": "Hello"}
    assert message_parts[1] == {"type": "text", "text": "World"}


def test_to_message_content_with_images():
    """Test converting mixed content to message format."""
    content = MultipartContent()
    content.add_text("Describe this:")
    content.add_image(Image.from_url("https://example.com/img.png"))
    
    message_parts = content.to_message_content()
    
    assert len(message_parts) == 2
    assert message_parts[0]["type"] == "text"
    assert message_parts[1]["type"] == "image_url"
    assert "image_url" in message_parts[1]


def test_get_text_only():
    """Test extracting text-only content."""
    content = MultipartContent()
    content.add_text("Line 1")
    content.add_image(Image.from_url("https://example.com/img.png"))
    content.add_text("Line 2")
    content.add_text("Line 3")
    
    text = content.get_text_only()
    
    assert "Line 1" in text
    assert "Line 2" in text
    assert "Line 3" in text
    assert text.count("\n") == 2  # Joined with newlines


def test_has_images_false():
    """Test has_images returns False for text-only content."""
    content = MultipartContent()
    content.add_text("Just text")
    
    assert not content.has_images()


def test_has_images_true():
    """Test has_images returns True when images present."""
    content = MultipartContent()
    content.add_text("Text")
    content.add_image(Image.from_url("https://example.com/img.png"))
    
    assert content.has_images()


def test_len():
    """Test length of MultipartContent."""
    content = MultipartContent()
    assert len(content) == 0
    
    content.add_text("Text")
    assert len(content) == 1
    
    content.add_image(Image.from_url("https://example.com/img.png"))
    assert len(content) == 2


def test_str_representation():
    """Test string representation."""
    content = MultipartContent()
    content.add_text("Text 1")
    content.add_text("Text 2")
    content.add_image(Image.from_url("https://example.com/img1.png"))
    content.add_image(Image.from_url("https://example.com/img2.png"))
    
    str_repr = str(content)
    assert "2 text parts" in str_repr
    assert "2 images" in str_repr


def test_init_with_parts():
    """Test initializing with parts list."""
    img = Image.from_url("https://example.com/img.png")
    content = MultipartContent(parts=["Hello", img, "World"])
    
    assert len(content) == 3
    assert content.has_images()
    assert isinstance(content.parts[1], Image)

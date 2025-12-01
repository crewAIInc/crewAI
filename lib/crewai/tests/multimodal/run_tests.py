#!/usr/bin/env python3
"""Simple test runner for multimodal implementation."""

import base64
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from crewai.multimodal.image import Image
from crewai.multimodal.multipart_content import MultipartContent


def test_image_from_url():
    """Test creating Image from URL."""
    print("Testing Image.from_url()...", end=" ")
    img = Image.from_url("https://example.com/image.png")
    assert img.source == "https://example.com/image.png"
    assert img.source_type == "url"
    assert img.media_type == "image/png"
    print("✓")


def test_image_from_file():
    """Test creating Image from file path."""
    print("Testing Image.from_file()...", end=" ")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"fake png data")
        tmp_path = tmp.name
    
    try:
        img = Image.from_file(tmp_path)
        assert img.source_type == "file"
        assert img.media_type == "image/png"
        print("✓")
    finally:
        Path(tmp_path).unlink()


def test_image_from_base64():
    """Test creating Image from base64 string."""
    print("Testing Image.from_base64()...", end=" ")
    b64_data = base64.b64encode(b"fake image data").decode("utf-8")
    img = Image.from_base64(b64_data, media_type="image/jpeg")
    
    assert img.source == b64_data
    assert img.source_type == "base64"
    assert img.media_type == "image/jpeg"
    print("✓")


def test_image_from_binary():
    """Test creating Image from binary data."""
    print("Testing Image.from_binary()...", end=" ")
    binary_data = b"fake binary image data"
    img = Image.from_binary(binary_data)
    
    assert img.source == binary_data
    assert img.source_type == "binary"
    assert img.media_type == "image/png"
    print("✓")


def test_image_from_placeholder():
    """Test creating Image placeholder."""
    print("Testing Image.from_placeholder()...", end=" ")
    img = Image.from_placeholder("user_image")
    
    assert img.placeholder == "user_image"
    assert img.media_type == "image/png"
    print("✓")


def test_image_auto_infer_source_type():
    """Test automatic source type inference."""
    print("Testing Image source type auto-inference...", end=" ")
    
    # URL
    img = Image(source="https://example.com/img.png")
    assert img.source_type == "url"
    
    # File path
    img = Image(source="/path/to/image.png")
    assert img.source_type == "file"
    
    # Data URL
    img = Image(source="data:image/png;base64,iVBORw0KGgo=")
    assert img.source_type == "data_url"
    
    # Base64 (long string without path separators)
    long_base64 = "A" * 200
    img = Image(source=long_base64)
    assert img.source_type == "base64"
    
    # Binary
    img = Image(source=b"binary data")
    assert img.source_type == "binary"
    
    print("✓")


def test_to_data_url():
    """Test converting various sources to data URL."""
    print("Testing to_data_url() conversions...", end=" ")
    
    # URL returns as-is
    img = Image.from_url("https://example.com/image.png")
    assert img.to_data_url() == "https://example.com/image.png"
    
    # Base64 conversion
    b64_data = base64.b64encode(b"test data").decode("utf-8")
    img = Image.from_base64(b64_data, media_type="image/jpeg")
    data_url = img.to_data_url()
    assert data_url.startswith("data:image/jpeg;base64,")
    assert b64_data in data_url
    
    # Binary conversion
    binary_data = b"test binary data"
    img = Image.from_binary(binary_data, media_type="image/png")
    data_url = img.to_data_url()
    expected_b64 = base64.b64encode(binary_data).decode("utf-8")
    assert data_url == f"data:image/png;base64,{expected_b64}"
    
    # File conversion
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        test_data = b"fake png file content"
        tmp.write(test_data)
        tmp_path = tmp.name
    
    try:
        img = Image.from_file(tmp_path)
        data_url = img.to_data_url()
        expected_b64 = base64.b64encode(test_data).decode("utf-8")
        assert data_url == f"data:image/png;base64,{expected_b64}"
    finally:
        Path(tmp_path).unlink()
    
    print("✓")


def test_to_message_content():
    """Test converting image to message content format."""
    print("Testing to_message_content()...", end=" ")
    img = Image.from_url("https://example.com/image.png")
    content = img.to_message_content()
    
    assert content["type"] == "image_url"
    assert "image_url" in content
    assert content["image_url"]["url"] == "https://example.com/image.png"
    print("✓")


def test_multipart_content_empty():
    """Test creating empty MultipartContent."""
    print("Testing empty MultipartContent...", end=" ")
    content = MultipartContent()
    assert len(content) == 0
    assert not content.has_images()
    print("✓")


def test_multipart_add_text_and_image():
    """Test adding text and images to MultipartContent."""
    print("Testing MultipartContent.add_text() and add_image()...", end=" ")
    content = MultipartContent()
    content.add_text("Hello world")
    content.add_text("Another line")
    
    assert len(content) == 2
    assert content.parts[0] == "Hello world"
    assert content.parts[1] == "Another line"
    
    img = Image.from_url("https://example.com/image.png")
    content.add_image(img)
    
    assert len(content) == 3
    assert content.has_images()
    assert isinstance(content.parts[2], Image)
    print("✓")


def test_multipart_to_message_content():
    """Test converting multipart content to message format."""
    print("Testing MultipartContent.to_message_content()...", end=" ")
    content = MultipartContent()
    content.add_text("Describe this:")
    content.add_image(Image.from_url("https://example.com/img.png"))
    
    message_parts = content.to_message_content()
    
    assert len(message_parts) == 2
    assert message_parts[0]["type"] == "text"
    assert message_parts[0]["text"] == "Describe this:"
    assert message_parts[1]["type"] == "image_url"
    assert "image_url" in message_parts[1]
    print("✓")


def test_multipart_get_text_only():
    """Test extracting text-only content."""
    print("Testing MultipartContent.get_text_only()...", end=" ")
    content = MultipartContent()
    content.add_text("Line 1")
    content.add_image(Image.from_url("https://example.com/img.png"))
    content.add_text("Line 2")
    content.add_text("Line 3")
    
    text = content.get_text_only()
    
    assert "Line 1" in text
    assert "Line 2" in text
    assert "Line 3" in text
    assert text.count("\n") == 2
    print("✓")


def test_agent_multipart_context():
    """Test that Agent accepts multipart_context parameter."""
    print("Testing Agent with multipart_context...", end=" ")
    try:
        from crewai.agent.core import Agent
        
        # Test with list of strings and images
        img = Image.from_url("https://example.com/test.png")
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            multipart_context=["Text context", img, "More text"]
        )
        
        assert agent.multipart_context is not None
        assert len(agent.multipart_context) == 3
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")
        raise


def test_task_multipart_context():
    """Test that Task accepts multipart_context parameter."""
    print("Testing Task with multipart_context...", end=" ")
    try:
        from crewai.task import Task
        
        # Test with MultipartContent
        content = MultipartContent()
        content.add_text("Analyze this image:")
        content.add_image(Image.from_url("https://example.com/scene.jpg"))
        
        task = Task(
            description="Describe the scene",
            expected_output="Detailed description",
            multipart_context=content
        )
        
        assert task.multipart_context is not None
        assert isinstance(task.multipart_context, MultipartContent)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")
        raise


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Multimodal Implementation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_image_from_url,
        test_image_from_file,
        test_image_from_base64,
        test_image_from_binary,
        test_image_from_placeholder,
        test_image_auto_infer_source_type,
        test_to_data_url,
        test_to_message_content,
        test_multipart_content_empty,
        test_multipart_add_text_and_image,
        test_multipart_to_message_content,
        test_multipart_get_text_only,
        test_agent_multipart_context,
        test_task_multipart_context,
    ]
    
    failed = []
    
    for test in tests:
        try:
            test()
        except Exception as e:
            failed.append((test.__name__, e))
            print(f"  Error: {e}")
    
    print()
    print("=" * 60)
    if not failed:
        print(f"✓ All {len(tests)} tests passed!")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {len(failed)} test(s) failed:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

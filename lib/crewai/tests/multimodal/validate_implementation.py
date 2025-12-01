#!/usr/bin/env python3
"""Standalone test for multimodal classes - tests Image and MultipartContent in isolation."""

import base64
import sys
import tempfile
from pathlib import Path

# Test just the multimodal module standalone
test_dir = Path(__file__).parent.parent.parent / "src" / "crewai" / "multimodal"

print(f"Testing multimodal module from: {test_dir}")
print(f"Module exists: {test_dir.exists()}")
print()

# Quick validation that our files have valid Python syntax
print("Validating Python syntax...")
for py_file in [test_dir / "image.py", test_dir / "multipart_content.py", test_dir / "__init__.py"]:
    print(f"  Checking {py_file.name}...", end=" ")
    try:
        with open(py_file) as f:
            compile(f.read(), py_file, 'exec')
        print("✓ Valid syntax")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        sys.exit(1)

print()
print("=" * 60)
print("All syntax checks passed!")
print("=" * 60)
print()

# Check that imports are structured correctly
print("Checking module structure...")
print("  Image class imports:", end=" ")
with open(test_dir / "image.py") as f:
    content = f.read()
    assert "class Image(BaseModel):" in content
    assert "from pydantic import" in content
    assert "def from_url(" in content
    assert "def from_file(" in content
    assert "def from_base64(" in content
    assert "def from_binary(" in content
    assert "def from_placeholder(" in content
    assert "def to_data_url(" in content
    assert "def to_message_content(" in content
    print("✓")

print("  MultipartContent class imports:", end=" ")
with open(test_dir / "multipart_content.py") as f:
    content = f.read()
    assert "class MultipartContent(BaseModel):" in content
    assert "from crewai.multimodal.image import Image" in content
    assert "def add_text(" in content
    assert "def add_image(" in content
    assert "def to_message_content(" in content
    assert "def get_text_only(" in content
    assert "def has_images(" in content
    print("✓")

print("  __init__.py exports:", end=" ")
with open(test_dir / "__init__.py") as f:
    content = f.read()
    assert "from crewai.multimodal.image import Image" in content
    assert "from crewai.multimodal.multipart_content import MultipartContent" in content
    assert '__all__ = ["Image", "MultipartContent"]' in content
    print("✓")

print()
print("=" * 60)
print("✓ Module structure validation passed!")
print("=" * 60)
print()

# Check Agent and Task modifications
print("Checking Agent and Task modifications...")
agent_file = Path(__file__).parent.parent.parent / "src" / "crewai" / "agent" / "core.py"
task_file = Path(__file__).parent.parent.parent / "src" / "crewai" / "task.py"

print(f"  Checking Agent ({agent_file.name})...", end=" ")
with open(agent_file) as f:
    content = f.read()
    assert "from crewai.multimodal import Image, MultipartContent" in content
    assert "multipart_context: list[str | Image] | MultipartContent | None" in content
    print("✓ multipart_context field added")

print(f"  Checking Task ({task_file.name})...", end=" ")
with open(task_file) as f:
    content = f.read()
    assert "from crewai.multimodal import Image, MultipartContent" in content
    assert "multipart_context: list[str | Image] | MultipartContent | None" in content
    print("✓ multipart_context field added")

print()
print("=" * 60)
print("✓ All integration checks passed!")
print("=" * 60)
print()

print("Summary:")
print("  ✓ Image class implemented with all required methods")
print("  ✓ MultipartContent class implemented with all required methods")
print("  ✓ Agent.multipart_context field added")
print("  ✓ Task.multipart_context field added")
print("  ✓ Proper imports and exports configured")
print()
print("The implementation is ready for testing with actual pydantic/crewai environment!")
print()
print("To run full unit tests once dependencies are installed:")
print("  cd lib/crewai && pytest tests/multimodal/ -v")

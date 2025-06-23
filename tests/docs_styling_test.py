import os
import pytest
from pathlib import Path

def test_custom_css_file_exists():
    """Test that the custom CSS file exists in the docs directory."""
    css_file = Path(__file__).parent.parent / "docs" / "style.css"
    assert css_file.exists(), "Custom CSS file should exist in docs directory"

def test_frame_component_styling():
    """Test that the CSS file contains proper Frame component styling."""
    css_file = Path(__file__).parent.parent / "docs" / "style.css"
    
    with open(css_file, 'r') as f:
        css_content = f.read()
    
    assert 'frame' in css_content.lower(), "CSS should contain frame component styling"
    assert 'min-width' in css_content, "CSS should contain min-width property"
    assert 'overflow-x' in css_content, "CSS should contain overflow-x property"

def test_installation_mdx_has_frame_component():
    """Test that the installation.mdx file contains the Frame component."""
    mdx_file = Path(__file__).parent.parent / "docs" / "installation.mdx"
    
    with open(mdx_file, 'r') as f:
        content = f.read()
    
    assert '<Frame>' in content, "installation.mdx should contain Frame component"
    assert 'my_project/' in content, "Frame should contain the file structure"

def test_css_contains_required_selectors():
    """Test that the CSS file contains all required selectors for Frame components."""
    css_file = Path(__file__).parent.parent / "docs" / "style.css"
    
    with open(css_file, 'r') as f:
        css_content = f.read()
    
    required_selectors = [
        '.frame-container',
        '[data-component="frame"]',
        '.frame',
        'div[class*="frame"]'
    ]
    
    for selector in required_selectors:
        assert selector in css_content, f"CSS should contain selector: {selector}"

def test_css_has_proper_width_properties():
    """Test that the CSS file has proper width and overflow properties."""
    css_file = Path(__file__).parent.parent / "docs" / "style.css"
    
    with open(css_file, 'r') as f:
        css_content = f.read()
    
    assert 'min-width: 300px' in css_content, "CSS should set min-width to 300px"
    assert 'width: 100%' in css_content, "CSS should set width to 100%"
    assert 'overflow-x: auto' in css_content, "CSS should set overflow-x to auto"
    assert 'white-space: pre' in css_content, "CSS should preserve whitespace in pre elements"

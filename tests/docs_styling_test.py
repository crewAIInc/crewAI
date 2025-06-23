import os
import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def css_file_path():
    """Fixture providing the path to the CSS file."""
    return Path(__file__).parent.parent / "docs" / "style.css"

@pytest.fixture(scope="module")
def css_content(css_file_path):
    """Fixture providing the CSS file content."""
    with open(css_file_path, 'r', encoding='utf-8') as f:
        return f.read()

@pytest.fixture(scope="module")
def mdx_file_path():
    """Fixture providing the path to the installation MDX file."""
    return Path(__file__).parent.parent / "docs" / "installation.mdx"

def test_custom_css_file_exists(css_file_path):
    """Test that the custom CSS file exists in the docs directory."""
    assert css_file_path.exists(), (
        f"Custom CSS file not found at {css_file_path}. "
        "Please ensure style.css is present in the docs directory."
    )

def test_frame_component_styling(css_content):
    """Test that the CSS file contains proper Frame component styling."""
    assert 'frame' in css_content.lower(), "CSS should contain frame component styling"
    assert 'min-width' in css_content, "CSS should contain min-width property"
    assert 'overflow-x' in css_content, "CSS should contain overflow-x property"

def test_installation_mdx_has_frame_component(mdx_file_path):
    """Test that the installation.mdx file contains the Frame component."""
    with open(mdx_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '<Frame>' in content, "installation.mdx should contain Frame component"
    assert 'my_project/' in content, "Frame should contain the file structure"

def test_css_contains_required_selectors(css_content):
    """Test that the CSS file contains all required selectors for Frame components."""
    required_selectors = [
        '.frame-container',
        '[data-component="frame"]',
        '.frame',
        'div[class*="frame"]'
    ]
    
    for selector in required_selectors:
        assert selector in css_content, f"CSS should contain selector: {selector}"

def test_css_has_proper_width_properties(css_content):
    """Test that the CSS file has proper width and overflow properties."""
    assert '--frame-min-width: 300px' in css_content, "CSS should define frame min-width variable"
    assert '--frame-width: 100%' in css_content, "CSS should define frame width variable"
    assert 'overflow-x: auto' in css_content, "CSS should set overflow-x to auto"
    assert 'white-space: pre' in css_content, "CSS should preserve whitespace in pre elements"

def test_css_values_are_valid(css_content):
    """Validate that critical CSS values are as expected."""
    assert '300px' in css_content, "Min-width should be 300px"
    assert '100%' in css_content, "Width should be 100%"
    assert 'var(--frame-min-width)' in css_content, "CSS should use custom properties"

def test_responsive_design_included(css_content):
    """Test that responsive design media queries are included."""
    assert '@media screen and (max-width: 768px)' in css_content, "CSS should include responsive media queries"

def test_vendor_prefixes_included(css_content):
    """Test that vendor prefixes are included for better browser compatibility."""
    assert '-webkit-overflow-scrolling: touch' in css_content, "CSS should include webkit overflow scrolling for iOS"

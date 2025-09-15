"""Test documentation link integrity to prevent broken links."""

import os
from pathlib import Path
import re
import pytest


def test_integration_overview_links():
    """Test that integration overview page links point to existing documentation files."""
    overview_file = Path(__file__).parent.parent / "docs" / "en" / "tools" / "tool-integrations" / "overview.mdx"
    
    with open(overview_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    href_pattern = r'href="(/en/tools/[^"]+)"'
    hrefs = re.findall(href_pattern, content)
    
    docs_root = Path(__file__).parent.parent / "docs"
    for href in hrefs:
        file_path = docs_root / href.lstrip('/') + '.mdx'
        assert file_path.exists(), f"Documentation file not found for href: {href}"


def test_specific_integration_links():
    """Test the specific links mentioned in issue #3516."""
    docs_root = Path(__file__).parent.parent / "docs"
    
    bedrock_file = docs_root / "en" / "tools" / "integration" / "bedrockinvokeagenttool.mdx"
    crewai_automation_file = docs_root / "en" / "tools" / "integration" / "crewaiautomationtool.mdx"
    
    assert bedrock_file.exists(), "Bedrock Invoke Agent Tool documentation file should exist"
    assert crewai_automation_file.exists(), "CrewAI Automation Tool documentation file should exist"
    
    overview_file = docs_root / "en" / "tools" / "tool-integrations" / "overview.mdx"
    with open(overview_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '/en/tools/integration/bedrockinvokeagenttool' in content, "Overview should link to correct Bedrock tool path"
    assert '/en/tools/integration/crewaiautomationtool' in content, "Overview should link to correct CrewAI automation tool path"
    
    assert '/en/tools/tool-integrations/bedrockinvokeagenttool' not in content, "Overview should not use incorrect Bedrock tool path"
    assert '/en/tools/tool-integrations/crewaiautomationtool' not in content, "Overview should not use incorrect CrewAI automation tool path"

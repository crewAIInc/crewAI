from __future__ import annotations

import ast
from pathlib import Path


DOCS_ROOT = Path(__file__).parents[4] / "docs"
MEMORY_DOC = DOCS_ROOT / "en" / "concepts" / "memory.mdx"


def test_memory_docs_document_agent_magnet_base_url_integration() -> None:
    """Document the Agent Magnet base URL path for cross-session memory."""
    content = MEMORY_DOC.read_text(encoding="utf-8")

    section_start = content.index("### Agent Magnet base URL integration")
    section_end = content.index("## Storage Backend", section_start)
    section = content[section_start:section_end]

    assert "Agent Magnet" in section
    assert "base_url" in section
    assert "OPENAI_BASE_URL" in section
    assert "agentmagnet.app" in section
    assert "https://magnet-gateway.onrender.com/v1" in section
    assert "x-session-id" in section
    assert "default_headers" in section
    assert "{question}" in section
    assert "Memory(" in section
    assert "llm=agent_magnet_llm" in section
    assert "memory=True" in section
    assert "root_scope" in section
    assert "scope" in section
    assert "without adding an Agent Magnet dependency" in section

    python_block = section.split("```python", 1)[1].split("```", 1)[0]
    ast.parse(python_block)

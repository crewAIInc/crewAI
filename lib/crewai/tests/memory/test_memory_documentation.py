from __future__ import annotations

import ast
from pathlib import Path


DOCS_ROOT = Path(__file__).parents[4] / "docs"
MEMORY_DOC = DOCS_ROOT / "edge" / "en" / "concepts" / "memory.mdx"


def test_memory_docs_document_custom_gateway_base_url_integration() -> None:
    """Keep the custom gateway memory snippet parseable and vendor-neutral."""
    content = MEMORY_DOC.read_text(encoding="utf-8")

    section_start = content.index("### Custom OpenAI-compatible gateway")
    section_end = content.index("## Storage Backend", section_start)
    section = content[section_start:section_end]

    assert "Agent Magnet" not in section
    assert "base_url" in section
    assert "OPENAI_BASE_URL" in section
    assert "onrender.com" not in section
    assert "your-gateway.example.com" in section
    assert "x-session-id" in section
    assert "default_headers" in section
    assert "{question}" in section
    assert "Memory(" in section
    assert "llm=gateway_llm" in section
    assert "memory=True" in section
    assert "root_scope" in section
    assert "scope" in section

    python_block = section.split("```python", 1)[1].split("```", 1)[0]
    ast.parse(python_block)

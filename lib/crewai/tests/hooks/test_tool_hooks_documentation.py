from __future__ import annotations

from pathlib import Path


DOCS_ROOT = Path(__file__).parents[4] / "docs"
TOOL_HOOKS_DOC = DOCS_ROOT / "en" / "learn" / "tool-hooks.mdx"


def test_tool_hooks_document_agent_threat_rules_integration_path() -> None:
    """Document the pre-tool hook path for Agent Threat Rules scanners."""
    content = TOOL_HOOKS_DOC.read_text(encoding="utf-8")

    section_start = content.index("#### Agent Threat Rules pre-tool scanner")
    section_end = content.index("### 2. Human Approval Gate", section_start)
    atr_section = content[section_start:section_end]

    assert "Agent Threat Rules" in atr_section
    assert "A recommended CrewAI integration path" in atr_section
    assert "@before_tool_call" in atr_section
    assert "scan_tool_intent_with_atr" in atr_section
    assert "scan_with_agent_threat_rules" in atr_section
    assert "context.tool_name" in atr_section
    assert "context.tool_input" in atr_section
    assert "context.agent" in atr_section
    assert "context.task" in atr_section
    assert "context.crew" in atr_section
    assert "agent_fingerprint" in atr_section
    assert "task_fingerprint" in atr_section
    assert "crew_fingerprint" in atr_section
    assert "return False" in atr_section

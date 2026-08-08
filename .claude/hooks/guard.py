#!/usr/bin/env python3
"""Block tool calls that violate a written contributor rule.

Wired as a Claude Code `PreToolUse` hook in `.claude/settings.json`. Every rule
here cites a rule already written down in `.github/CONTRIBUTING.md` or
`AGENTS.md` — this file enforces those documents, it does not add policy of its
own. A rule that is not written down there does not belong here.

Reads the hook event as JSON on stdin, writes a JSON decision on stdout, and
exits 0 either way. No subprocess, no filesystem writes, no network.

Escape hatch: include `# policy-override: <reason>` in a Bash command to state
an exception explicitly rather than working around the guard silently.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


OVERRIDE_MARKER = "# policy-override:"

#: (pattern, reason). Patterns match the raw Bash command string.
BASH_RULES: tuple[tuple[str, str], ...] = (
    (
        r"(?:^|[;&|(\n])\s*(?:sudo\s+)?(?:pip3?|python3?\s+-m\s+pip)\s+"
        r"(?:install|uninstall)\b",
        "CONTRIBUTING.md (Dependency Management): do not use pip directly. "
        "Use `uv add --package <pkg> <dep>`, `uv add --dev <dep>`, or `uv sync`.",
    ),
    (
        r"\bgit\b[^\n;&|]*\b(?:commit|push)\b[^\n;&|]*(?:--no-verify\b|\s-n(?=\s|$))",
        "CONTRIBUTING.md (Commits): do not use --no-verify to skip hooks. "
        "Fix what pre-commit reports instead.",
    ),
    (
        r"\b(?:rm|mv)\b[^\n]*\bdocs/images/",
        "AGENTS.md (Changing Docs, rule 3): do not delete or rename files under "
        "docs/images/ — frozen doc snapshots still reference them.",
    ),
    (
        r"\b(?:rm|mv|tee)\b[^\n]*\bdocs/v[0-9]",
        "AGENTS.md (Changing Docs, rule 2): docs/v*/ are frozen release snapshots "
        "managed by devtools. Edit docs/edge/en/ instead.",
    ),
)

FROZEN_DOCS_REASON = (
    "AGENTS.md (Changing Docs, rule 2): docs/v*/ are frozen release snapshots "
    "managed by devtools. Edit the MDX under docs/edge/en/ instead, then sync the "
    "ar, ko, and pt-BR translations."
)


def deny(reason: str) -> None:
    """Emit a deny decision.

    The T201 suppression is deliberate and not a lint escape: stdout is the hook
    protocol itself — Claude Code parses this JSON to decide whether to proceed.
    """
    print(  # noqa: T201
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def bash_violation(command: str) -> str | None:
    """Return the reason this command is blocked, or None if it is allowed."""
    if OVERRIDE_MARKER in command:
        return None
    for pattern, reason in BASH_RULES:
        if re.search(pattern, command):
            return reason
    return None


def edits_frozen_docs(path: str) -> bool:
    """True when path targets a frozen release snapshot under docs/v<digit>."""
    match = re.search(r"(?:^|/)docs/v[0-9]", path)
    return match is not None


def target_path(tool_input: dict[str, Any]) -> str:
    """The file a write-shaped tool is about to touch, or an empty string."""
    for key in ("file_path", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(event, dict):
        return

    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return

    command = tool_input.get("command")
    if isinstance(command, str):
        reason = bash_violation(command)
        if reason is not None:
            deny(reason)
        return

    if edits_frozen_docs(target_path(tool_input)):
        deny(FROZEN_DOCS_REASON)


if __name__ == "__main__":
    main()

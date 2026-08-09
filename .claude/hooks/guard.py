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
        # Version-suffixed interpreters (pip3.12, python3.12 -m pip) bypass a bare
        # pip3? match. `uv pip` is unaffected: the anchor requires command position.
        r"(?:^|[;&|(\n])\s*(?:sudo\s+)?"
        r"(?:pip(?:[23](?:\.\d+)?)?|python(?:[23](?:\.\d+)?)?\s+-m\s+pip)\s+"
        r"(?:install|uninstall)\b",
        "CONTRIBUTING.md (Dependency Management): do not use pip directly. "
        "Use `uv add --package <pkg> <dep>`, `uv add --dev <dep>`, or `uv sync`.",
    ),
    (
        # `-n` is only the skip-hooks flag for commit; on push it means --dry-run,
        # which is safe and must stay allowed.
        r"\bgit\b[^\n;&|]*\b(?:commit|push)\b[^\n;&|]*--no-verify\b"
        r"|\bgit\b[^\n;&|]*\bcommit\b[^\n;&|]*\s-n(?=\s|$)",
        "CONTRIBUTING.md (Commits): do not use --no-verify to skip hooks. "
        "Fix what pre-commit reports instead.",
    ),
    (
        # Stops at shell separators so a later segment merely naming the path does
        # not trigger, and matches the directory with or without a trailing slash.
        r"\b(?:rm|mv)\b[^\n;&|]*\bdocs/images(?![\w.-])",
        "AGENTS.md (Changing Docs, rule 3): do not delete or rename files under "
        "docs/images/ — frozen doc snapshots still reference them.",
    ),
    (
        # Write verbs and output redirection. Read-only access (cat, grep, less)
        # is deliberately allowed. `cp` is matched in either direction: failing
        # closed on a copy out of the tree is cheaper than missing a copy into it.
        r"\b(?:rm|mv|cp|tee|sed\s+-i)\b[^\n;&|]*\bdocs/v[0-9]"
        r"|>>?\s*[^\n;&|]*\bdocs/v[0-9]",
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


def strip_heredocs(command: str) -> str:
    """Replace heredoc bodies with a placeholder.

    A heredoc body is data — a commit message, a file being written, a PR body —
    not a command. Matching rules against it produces false denials, such as
    blocking a commit whose message merely discusses `rm -rf docs/images`. The
    redirection and the delimiter stay on the command line, so a heredoc that
    genuinely writes into a protected path is still caught.
    """
    return re.sub(
        r"(<<-?\s*[\"']?(\w+)[\"']?).*?^\s*\2\b",
        r"\1 HEREDOC_BODY",
        command,
        flags=re.DOTALL | re.MULTILINE,
    )


def bash_violation(command: str) -> str | None:
    """Return the reason this command is blocked, or None if it is allowed."""
    if OVERRIDE_MARKER in command:
        return None
    inspectable = strip_heredocs(command)
    for pattern, reason in BASH_RULES:
        if re.search(pattern, inspectable):
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

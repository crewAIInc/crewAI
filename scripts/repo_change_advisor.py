#!/usr/bin/env python3
"""Analyze local repository changes with a command-capable CrewAI agent."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from crewai import Agent, Crew, LLM, Task
from crewai.tools import tool

REPO_ROOT = Path.cwd().resolve()

SAFE_GIT_SUBCOMMANDS = {
    "status",
    "log",
    "diff",
    "show",
    "rev-parse",
    "branch",
    "remote",
    "ls-files",
    "cat-file",
    "blame",
}

SAFE_NON_GIT = {"rg", "ls", "cat", "sed", "head", "tail", "wc", "pwd", "find"}


def _targets_codex_model(model: str) -> bool:
    value = model.strip().lower()
    return "codex" in value


def _codex_login_status() -> tuple[bool, str]:
    """Return (logged_in, message) from `codex login status`."""
    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False, "codex CLI not found in PATH"
    except Exception as exc:  # noqa: BLE001
        return False, f"codex login status failed: {exc}"

    msg = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return proc.returncode == 0, (msg or f"exit_code={proc.returncode}")


def _configure_auth(model: str) -> tuple[str, str]:
    """Configure auth based on model family and local login state."""
    needs_codex_route = _targets_codex_model(model)
    logged_in, login_message = _codex_login_status()
    if needs_codex_route and logged_in:
        os.environ["CREWAI_OPENAI_AUTH_MODE"] = "oauth_codex"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_OAUTH_ACCESS_TOKEN", None)
        os.environ.pop("OPENAI_ACCESS_TOKEN", None)
        return "codex_oauth", login_message

    os.environ.pop("CREWAI_OPENAI_AUTH_MODE", None)
    if not os.getenv("OPENAI_API_KEY"):
        print("auth_strategy=api_key_required")
        print(f"codex_login_status={login_message}")
        if needs_codex_route:
            print("ERROR: codex model selected, but Codex OAuth is unavailable and OPENAI_API_KEY is not set.")
            print("Set OPENAI_API_KEY, or run `codex login`, then retry.")
        else:
            print("ERROR: non-codex model requires OPENAI_API_KEY in this script.")
            print("Set OPENAI_API_KEY, or switch to a codex model and run `codex login`.")
        raise SystemExit(2)
    return "api_key", login_message


def _is_safe_command(command: str) -> tuple[bool, str]:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return False, f"invalid command: {exc}"

    if not tokens:
        return False, "empty command"

    cmd = tokens[0]
    if cmd == "git":
        if len(tokens) < 2:
            return False, "git subcommand required"
        sub = tokens[1]
        if sub not in SAFE_GIT_SUBCOMMANDS:
            return False, f"git subcommand not allowed: {sub}"
        return True, ""

    if cmd in SAFE_NON_GIT:
        return True, ""

    return False, f"command not allowed: {cmd}"


def _normalize_repo_scoped_command(command: str) -> tuple[str | None, str]:
    """Allow `cd <repo> && ...` patterns by normalizing to repo-local command."""
    match = re.match(r"^\s*cd\s+(.+?)\s*&&\s*(.+?)\s*$", command)
    if not match:
        return command, ""

    raw_path = match.group(1).strip().strip("'\"")
    remainder = match.group(2).strip()
    try:
        target = Path(raw_path).expanduser().resolve()
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid cd path: {exc}"

    if target != REPO_ROOT:
        return None, f"cd target not allowed: {target}"
    return remainder, ""


def _split_command_chain(command: str) -> list[str]:
    """Split `a && b && c` into commands."""
    return [segment.strip() for segment in command.split("&&") if segment.strip()]


@tool("run_repo_command")
def run_repo_command(command: str, timeout_sec: int = 20) -> str:
    """Run a read-only shell command in the current repository."""
    normalized, normalize_error = _normalize_repo_scoped_command(command)
    if normalized is None:
        return f"Denied: {normalize_error}"

    commands = _split_command_chain(normalized)
    if not commands:
        return "Denied: empty command"

    all_outputs: list[str] = []
    for single in commands:
        ok, reason = _is_safe_command(single)
        if not ok:
            all_outputs.append(f"$ {single}\n\nDenied: {reason}")
            break

        try:
            proc = subprocess.run(
                shlex.split(single),
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1, min(timeout_sec, 120)),
            )
        except Exception as exc:  # noqa: BLE001
            all_outputs.append(f"$ {single}\n\nCommand failed: {exc}")
            break

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        parts = [f"$ {single}", f"exit_code={proc.returncode}"]
        if stdout:
            parts.append("stdout:\n" + stdout)
        if stderr:
            parts.append("stderr:\n" + stderr)
        all_outputs.append("\n\n".join(parts))
        if proc.returncode != 0:
            break

    return "\n\n---\n\n".join(all_outputs)


@tool("read_repo_file")
def read_repo_file(path: str, max_chars: int = 12000) -> str:
    """Read a repository file safely with truncation."""
    try:
        p = (REPO_ROOT / path).resolve()
        p.relative_to(REPO_ROOT)
    except Exception:  # noqa: BLE001
        return f"Path outside repository is not allowed: {path}"

    if not p.exists() or not p.is_file():
        return f"File not found: {p}"

    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]..."
    return f"# {p}\n\n{text}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Repository path.")
    parser.add_argument(
        "--model",
        default="openai-codex/gpt-5.3-codex",
        help="Model in provider/model format.",
    )
    parser.add_argument(
        "--prompt",
        default="分析当前仓库的提交历史和工作区修改，给出风险点与改进建议。",
        help="Your analysis request.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        print(f"Invalid repo path: {repo_root}")
        return 2

    global REPO_ROOT
    REPO_ROOT = repo_root

    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    auth_strategy, login_message = _configure_auth(args.model)

    llm = LLM(model=args.model, api="responses", is_litellm=False)
    client_params = llm._get_client_params()
    auth_source = getattr(getattr(llm, "_resolved_openai_auth", None), "source", None)

    print(f"repo={REPO_ROOT}")
    print(f"auth_strategy={auth_strategy}")
    print(f"codex_login_status={login_message}")
    print(f"model={llm.model}")
    print(f"provider={llm.provider}")
    print(f"auth_source={auth_source}")
    print(f"base_url={client_params.get('base_url')}")

    analyst = Agent(
        role="Repository Change Reviewer",
        goal="Audit repository changes and produce practical, testable recommendations.",
        backstory=(
            "You are a senior engineer focused on change risk, regressions, and missing tests. "
            "You inspect git history, working tree diffs, and concrete code sections before concluding."
        ),
        llm=llm,
        tools=[run_repo_command, read_repo_file],
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description=(
            "Repository path: {repo_path}\n"
            "User request: {user_prompt}\n\n"
            "You MUST run commands first, not guess. At minimum run:\n"
            "1) git status --short --branch\n"
            "2) git log --oneline -n 12\n"
            "3) git diff --stat\n"
            "4) git diff --cached --stat\n"
            "5) git diff\n"
            "6) git diff --cached\n\n"
            "Then read key changed files with read_repo_file and provide:\n"
            "- Change summary\n"
            "- High-risk issues and regression points\n"
            "- Missing tests / validation gaps\n"
            "- Commit quality suggestions\n"
            "- A prioritized next-step checklist"
        ),
        expected_output=(
            "A concise markdown review with concrete findings and actionable next steps."
        ),
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=True)
    result = crew.kickoff(inputs={"repo_path": str(REPO_ROOT), "user_prompt": args.prompt})

    print("\n===== REVIEW RESULT =====")
    print(str(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Measure what coding agents learn from the AGENTS.md that `crewai create` ships.

Each question in the item bank is asked in a fresh, tool-less coding-agent session
(Claude Code and Codex) inside a directory holding only the instruction files, once
with this branch's AGENTS.md and once with the base branch's. Answers are graded
by the item's answer key. The command fails when either agent's score drops more
than ``--max-drop`` below the base.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Final, Literal, NotRequired, TypedDict

import click
from rich.console import Console
from rich.table import Table


console = Console()

Grade = Literal["right", "partly", "wrong"]
Agent = Literal["claude", "codex"]
Ask = Callable[[Path, str], str]

ITEMS_FILE: Final = Path(__file__).parent / "agents_md_items.json"
TEMPLATES: Final = "lib/cli/src/crewai_cli/templates"
POINTS: Final[dict[Grade, float]] = {"right": 1.0, "partly": 0.5, "wrong": 0.0}
# Every tool disabled, no MCP servers, only the project's settings: answers come
# from the instruction files and the model.
CLAUDE_CMD: Final = [
    "claude", "-p", "--model", "haiku", "--tools", "", "--setting-sources", "project",
    "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
]  # fmt: skip
CODEX_EXEC: Final = [
    "codex",
    "exec",
    "--skip-git-repo-check",
    "--ephemeral",
    "-s",
    "read-only",
    "--json",
]
# Neither agent needs the other's key; Codex reads its own from CODEX_HOME.
API_KEYS: Final = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
# Codex can't run without tools, so the question says not to use them and the
# sandbox is read-only.
NO_COMMANDS: Final = (
    "\n\nAnswer from what you know and this project's instructions; don't run commands."
)
EVALUATION_START: Final = "<!-- crewai-eval:start -->"
EVALUATION_END: Final = "<!-- crewai-eval:end -->"
# A trap counts only when the answer recommends it, not when it warns against it,
# before ("don't run `crewai update`") or after ("`crewai update` won't do this").
NEGATION: Final = re.compile(
    r"\b(don'?t|do not|never|not|won'?t|isn'?t|avoid|instead of|rather than|no need to)\b",
    re.I,
)
AFTER_NEGATION: Final = re.compile(
    r"[`'\"]?\s*(won'?t|isn'?t|doesn'?t|does not|is not|is something else|migrates)\b",
    re.I,
)


class Item(TypedDict):
    """One fact the file should teach, asked as a question and graded by its key."""

    id: str
    item: str
    q: str
    key: list[str]
    partial: NotRequired[list[str]]
    trap: NotRequired[list[str]]


def load_items(path: Path = ITEMS_FILE) -> list[Item]:
    """Read the item bank."""
    items: list[Item] = json.loads(path.read_text(encoding="utf-8"))
    return items


def grade(item: Item, answer: str) -> Grade:
    """Grade an answer: a recommended trap is wrong, every key right, any partial partly."""

    def hit(pattern: str) -> bool:
        return re.search(pattern, answer, re.I | re.S) is not None

    def recommends(pattern: str) -> bool:
        return any(
            not NEGATION.search(answer[max(0, m.start() - 60) : m.start()])
            and not AFTER_NEGATION.match(answer[m.end() : m.end() + 50])
            for m in re.finditer(pattern, answer, re.I | re.S)
        )

    if any(recommends(p) for p in item.get("trap", [])):
        return "wrong"
    if all(hit(p) for p in item["key"]):
        return "right"
    if any(hit(p) for p in item.get("partial", [])):
        return "partly"
    return "wrong"


def shipped_text(template: str) -> str:
    """The AGENTS.md a crew or flow project receives: evaluation blocks kept, markers dropped.

    Mirrors ``crewai_cli.utils._assistant_instructions(text, evaluation=True)``; a test
    holds the two to the same output.
    """
    return "".join(
        line
        for line in template.splitlines(keepends=True)
        if line.strip() not in (EVALUATION_START, EVALUATION_END)
    )


def read_at(ref: str | None, path: str) -> str:
    """A repo file's text at a git ref, or in the working tree when ``ref`` is None."""
    if ref is None:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607 - git from PATH, fixed argv
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return (Path(root) / path).read_text(encoding="utf-8")
    return subprocess.run(  # noqa: S603 - fixed argv; ref is a git ref the caller passed
        ["git", "show", f"{ref}:{path}"],  # noqa: S607 - git from PATH
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def write_project(dest: Path, ref: str | None) -> None:
    """Write the instruction files a new project gets at ``ref`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "AGENTS.md").write_text(
        shipped_text(read_at(ref, f"{TEMPLATES}/AGENTS.md")), encoding="utf-8"
    )
    (dest / "CLAUDE.md").write_text(
        read_at(ref, f"{TEMPLATES}/CLAUDE.md"), encoding="utf-8"
    )


def ask_claude(project: Path, question: str) -> str:
    """One Claude Code session with every tool disabled, so answers come from the files."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CLAUDE_CODE") and k not in ("CLAUDECODE", "OPENAI_API_KEY")
    }
    proc = subprocess.run(  # noqa: S603 - fixed argv; the question goes on stdin
        CLAUDE_CMD,
        input=question,
        capture_output=True,
        text=True,
        cwd=project,
        env=env,
        timeout=300,
        check=False,
    )
    return proc.stdout.strip()


def codex_answer(events: str) -> str:
    """The agent's messages from ``codex exec --json`` output."""
    messages = []
    for line in events.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        item = event.get("item") or {}
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "agent_message"
        ):
            messages.append(str(item.get("text") or ""))
    return "\n".join(messages)


def codex_asker(codex_home: Path) -> Ask:
    """Codex sessions logged in with ``OPENAI_API_KEY``, in a home of their own."""
    codex_home.mkdir(
        parents=True, exist_ok=True
    )  # codex refuses a CODEX_HOME that doesn't exist
    subprocess.run(
        ["codex", "login", "--with-api-key"],  # noqa: S607 - CLI from PATH
        input=os.environ["OPENAI_API_KEY"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "CODEX_HOME": str(codex_home)},
    )

    env = {k: v for k, v in os.environ.items() if k not in API_KEYS}
    env["CODEX_HOME"] = str(codex_home)

    def ask(project: Path, question: str) -> str:
        proc = subprocess.run(  # noqa: S603 - the question is one argv entry, never a shell
            [*CODEX_EXEC, question + NO_COMMANDS],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=project,
            timeout=300,
            check=False,
            env=env,
        )
        return codex_answer(proc.stdout)

    return ask


def score(
    ask: Ask, project: Path, items: Sequence[Item], reps: int, workers: int
) -> dict[str, float]:
    """Average points per item over ``reps`` sessions."""
    jobs = [item for item in items for _ in range(reps)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        grades = list(pool.map(lambda item: grade(item, ask(project, item["q"])), jobs))
    totals: dict[str, float] = {}
    for item, result in zip(jobs, grades, strict=True):
        totals[item["id"]] = totals.get(item["id"], 0.0) + POINTS[result] / reps
    return totals


def dropped(base: float, head: float, max_drop: float) -> bool:
    """Whether ``head`` is more than ``max_drop`` (a fraction) below ``base``."""
    return head < base * (1 - max_drop)


@click.command("agents-md-eval")
@click.option(
    "--base",
    default="origin/main",
    show_default=True,
    help="Git ref whose AGENTS.md is the baseline.",
)
@click.option(
    "--agent",
    "agents",
    type=click.Choice(["claude", "codex"]),
    multiple=True,
    default=("claude", "codex"),
    show_default=True,
)
@click.option("--reps", default=1, show_default=True, help="Sessions per question.")
@click.option("--workers", default=8, show_default=True, help="Sessions at once.")
@click.option(
    "--max-drop",
    default=0.10,
    show_default=True,
    help="Largest allowed score drop, as a fraction.",
)
def agents_md_eval(
    base: str, agents: tuple[Agent, ...], reps: int, workers: int, max_drop: float
) -> None:
    """Fail when this branch's AGENTS.md teaches coding agents less than BASE's."""
    items = load_items()
    failed = False
    table = Table("Agent", "Base", "This branch", "Change")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_project(root / "base", base)
        write_project(root / "head", None)
        askers: dict[Agent, Ask] = {"claude": ask_claude}
        if "codex" in agents:
            askers["codex"] = codex_asker(root / "codex-home")
        for agent in agents:
            before = score(askers[agent], root / "base", items, reps, workers)
            after = score(askers[agent], root / "head", items, reps, workers)
            old, new = sum(before.values()), sum(after.values())
            change = (new - old) / old if old else 0.0
            failed |= dropped(old, new, max_drop)
            table.add_row(agent, f"{old:.1f}", f"{new:.1f}", f"{change:+.1%}")
            lost = [i for i in items if after[i["id"]] < before[i["id"]] - 0.5]
            for item in lost:
                console.print(
                    f"[yellow]{agent} lost {item['id']}:[/yellow] {item['item']}"
                )
    console.print(table)
    if failed:
        console.print(
            f"[red]A score dropped more than {max_drop:.0%} below {base}.[/red]"
        )
        raise SystemExit(1)

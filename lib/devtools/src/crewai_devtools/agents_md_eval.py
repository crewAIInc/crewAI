"""Measure what coding agents learn from the AGENTS.md that `crewai create` ships.

Each question in the item bank is asked in a fresh, tool-less coding-agent session
(Claude Code or Codex, per model) inside a directory holding only the instruction
files, once with this branch's AGENTS.md and once with the base branch's. Answers
are graded by the item's answer key. The command fails when any model's score drops
more than ``--max-drop`` below the base, or when too many sessions give no answer
for the comparison to mean anything.
"""

from __future__ import annotations

from collections import Counter
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
from rich.markup import escape
from rich.table import Table


console = Console()

Grade = Literal["right", "partly", "wrong"]
Agent = Literal["claude", "codex"]
# One session: the agent's answer; raises NoAnswer (or TimeoutExpired) when there is none.
Ask = Callable[[Path, str], str]

ITEMS_FILE: Final = Path(__file__).parent / "agents_md_items.json"
TEMPLATES: Final = "lib/cli/src/crewai_cli/templates"
POINTS: Final[dict[Grade, float]] = {"right": 1.0, "partly": 0.5, "wrong": 0.0}
MODELS: Final = (
    "claude:claude-haiku-5-5",
    "claude:claude-sonnet-5-5",
    "codex:gpt-5.6-luna",
    "codex:gpt-5.6-sol",
)
SESSION_TIMEOUT: Final = 300
# Past this share of unanswered sessions, a score says more about the agent's
# availability than about the file.
MAX_UNANSWERED: Final = 0.02
# Every tool disabled, no MCP servers, only the project's settings: answers come
# from the instruction files and the model.
CLAUDE_CMD: Final = [
    "claude", "-p", "--tools", "", "--setting-sources", "project",
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
# Answers use straight or typographic (U+2019) apostrophes.
NEGATION: Final = re.compile(
    r"\b(don['\u2019]?t|doesn['\u2019]?t|didn['\u2019]?t|aren['\u2019]?t|can['\u2019]?t|do not|never|not"
    r"|won['\u2019]?t|isn['\u2019]?t|avoid|instead of|rather than|no need to"
    r"|replac(es|ed|ing)|deprecated|legacy)\b",
    re.I,
)
AFTER_NEGATION: Final = re.compile(
    r"[`'\"]?\s*(won['\u2019]?t|isn['\u2019]?t|doesn['\u2019]?t|does not|is not|is something else|is a different"
    r"|does something different|migrates)\b",
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


class NoAnswer(Exception):  # noqa: N818 - a reason, not an error the caller handles as one
    """A session that ended without an answer; the message says why."""


SECRET: Final = re.compile(r"sk-[A-Za-z0-9_-]+")
# Claude Code can end its output with a JSON log line ({"model": ..., "query_source": ...})
# after the message that says what went wrong.
LOG_LINE: Final = re.compile(r"^\{.*\}$")


def _lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def session_answer(proc: subprocess.CompletedProcess[str], text: str) -> str:
    """The session's answer, or NoAnswer naming its exit code and last error line."""
    if proc.returncode != 0:
        # Claude Code prints some failures ("Not logged in", "Invalid API key") on stdout.
        stderr, stdout = _lines(proc.stderr), _lines(proc.stdout)
        messages = [line for line in stderr if not LOG_LINE.match(line)] or [
            line for line in stdout if not LOG_LINE.match(line)
        ]
        line = (messages or stderr or stdout or ["no error output"])[-1]
        raise NoAnswer(SECRET.sub("sk-***", f"exit {proc.returncode}: {line[:160]}"))
    if not text:
        raise NoAnswer("empty answer")
    return text


def claude_asker(model: str) -> Ask:
    """Claude Code sessions with every tool disabled, so answers come from the files."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CLAUDE_CODE") and k not in ("CLAUDECODE", "OPENAI_API_KEY")
    }

    def ask(project: Path, question: str) -> str:
        proc = subprocess.run(  # noqa: S603 - fixed argv; the question goes on stdin
            [*CLAUDE_CMD, "--model", model],
            input=question,
            capture_output=True,
            text=True,
            cwd=project,
            env=env,
            timeout=SESSION_TIMEOUT,
            check=False,
        )
        return session_answer(proc, proc.stdout.strip())

    return ask


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


def codex_asker(codex_home: Path, model: str) -> Ask:
    """Codex sessions logged in with ``OPENAI_API_KEY``, in a home of their own."""
    codex_home.mkdir(parents=True, exist_ok=True)  # codex refuses a missing CODEX_HOME
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
            [*CODEX_EXEC, "-m", model, question + NO_COMMANDS],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=project,
            timeout=SESSION_TIMEOUT,
            check=False,
            env=env,
        )
        return session_answer(proc, codex_answer(proc.stdout))

    return ask


def attempt(ask: Ask, project: Path, question: str) -> tuple[str | None, str]:
    """One session: its answer, or None and why there was none."""
    try:
        return ask(project, question), ""
    except subprocess.TimeoutExpired:
        return None, f"timed out after {SESSION_TIMEOUT}s"
    except NoAnswer as e:
        return None, str(e)


def answer(ask: Ask, project: Path, question: str) -> tuple[str | None, str]:
    """The session's answer with one retry, or None and why the last attempt gave none."""
    text, reason = attempt(ask, project, question)
    return (text, reason) if text is not None else attempt(ask, project, question)


def score(
    ask: Ask, project: Path, items: Sequence[Item], reps: int, workers: int
) -> tuple[dict[str, float], Counter[str]]:
    """Average points per item over ``reps`` sessions, and why sessions gave no answer.

    A session with no answer scores zero and is counted by reason, so the caller can
    tell a file that teaches less from an agent that wasn't answering.
    """
    jobs = [item for item in items for _ in range(reps)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda item: answer(ask, project, item["q"]), jobs))
    totals: dict[str, float] = {}
    missing: Counter[str] = Counter()
    for item, (text, reason) in zip(jobs, answers, strict=True):
        if text is None:
            missing[reason] += 1
        points = POINTS[grade(item, text)] if text is not None else 0.0
        totals[item["id"]] = totals.get(item["id"], 0.0) + points / reps
    return totals, missing


def dropped(base: float, head: float, max_drop: float) -> bool:
    """Whether ``head`` is more than ``max_drop`` (a fraction) below ``base``."""
    return head < base * (1 - max_drop)


def parse_model(spec: str) -> tuple[Agent, str]:
    """Split ``claude:<model>`` or ``codex:<model>``."""
    cli, _, model = spec.partition(":")
    if cli == "claude" and model:
        return "claude", model
    if cli == "codex" and model:
        return "codex", model
    raise click.BadParameter(f"{spec!r}: use claude:<model> or codex:<model>")


@click.command("agents-md-eval")
@click.option(
    "--base",
    default="origin/main",
    show_default=True,
    help="Git ref whose AGENTS.md is the baseline.",
)
@click.option(
    "--head",
    default=None,
    help="Git ref whose AGENTS.md is evaluated. Default: the working tree.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    default=MODELS,
    show_default=True,
    help="Agent and model to ask, as claude:<model> or codex:<model>. Repeatable.",
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
    base: str,
    head: str | None,
    models: tuple[str, ...],
    reps: int,
    workers: int,
    max_drop: float,
) -> None:
    """Fail when this branch's AGENTS.md teaches coding agents less than BASE's."""
    specs = [parse_model(spec) for spec in models]
    items = load_items()
    sessions = len(items) * reps
    problems: list[str] = []
    table = Table("Model", "Base", "This branch", "Change", "No answer (base / branch)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_project(root / "base", base)
        write_project(root / "head", head)
        for cli, model in specs:
            label = f"{cli}:{model}"
            ask = (
                claude_asker(model)
                if cli == "claude"
                else codex_asker(root / "codex-home", model)
            )
            console.print(f"Asking {label}: {sessions} sessions per file")
            before, base_missing = score(ask, root / "base", items, reps, workers)
            after, head_missing = score(ask, root / "head", items, reps, workers)
            old, new = sum(before.values()), sum(after.values())
            change = (new - old) / old if old else 0.0
            base_n, head_n = base_missing.total(), head_missing.total()
            table.add_row(
                label,
                f"{old:.1f}",
                f"{new:.1f}",
                f"{change:+.1%}",
                f"{base_n} / {head_n}",
            )
            for file, reasons in (("base", base_missing), ("branch", head_missing)):
                if reasons:
                    why = "; ".join(
                        f"{reason} ({n})" for reason, n in reasons.most_common(3)
                    )
                    # Reasons and items are data: `[Errno 8]` and `[tool.crewai]` aren't markup.
                    console.print(
                        f"[yellow]{label} gave no answer with the {file} file: {escape(why)}[/yellow]"
                    )
            if max(base_n, head_n) > sessions * MAX_UNANSWERED:
                problems.append(
                    f"{label}: too many sessions gave no answer ({base_n} / {head_n} of {sessions})"
                    ", so the comparison isn't reliable."
                )
            elif dropped(old, new, max_drop):
                problems.append(
                    f"{label}: scored {change:.1%} against {base}, past the {max_drop:.0%} limit."
                )
            for item in items:
                if after[item["id"]] < before[item["id"]] - 0.5:
                    console.print(
                        f"[yellow]{label} lost {item['id']}:[/yellow] {escape(item['item'])}"
                    )
    console.print(table)
    for problem in problems:
        console.print(f"[red]{escape(problem)}[/red]")
    if problems:
        raise SystemExit(1)

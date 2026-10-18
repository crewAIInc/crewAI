from pathlib import Path
import re
import subprocess

from click.testing import CliRunner
import pytest

from crewai_devtools import agents_md_eval as ev


ITEM: ev.Item = {
    "id": "X-001",
    "item": "Update the CLI with uv tool install crewai --upgrade.",
    "q": "How do I update the CLI?",
    "key": [r"uv tool install crewai", r"--upgrade"],
    "partial": [r"uv tool"],
    "trap": [r"crewai update"],
}


def test_bank_items_are_well_formed():
    items = ev.load_items()
    ids = [item["id"] for item in items]
    assert len(items) >= 200
    assert len(ids) == len(set(ids))
    for item in items:
        assert item["q"].strip() and item["item"].strip() and item["key"], item["id"]
        for pattern in item["key"] + item.get("partial", []) + item.get("trap", []):
            re.compile(pattern)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Run `uv tool install crewai --upgrade`.", "right"),
        ("Use uv tool to reinstall it.", "partly"),
        ("Reinstall it with pip.", "wrong"),
        ("Run `crewai update`, then `uv tool install crewai --upgrade`.", "wrong"),
        ("Don't run `crewai update`; use `uv tool install crewai --upgrade`.", "right"),
        (
            "`crewai update` won't do it: run `uv tool install crewai --upgrade`.",
            "right",
        ),
    ],
)
def test_grade(answer, expected):
    assert ev.grade(ITEM, answer) == expected


def test_shipped_text_matches_what_crewai_create_writes():
    from crewai_cli.utils import _assistant_instructions

    template = ev.read_at(None, f"{ev.TEMPLATES}/AGENTS.md")
    assert ev.shipped_text(template) == _assistant_instructions(
        template, evaluation=True
    )
    assert ev.EVALUATION_START not in ev.shipped_text(template)


@pytest.mark.parametrize(
    ("base", "head", "fails"),
    [
        (200.0, 180.0, False),
        (200.0, 179.9, True),
        (200.0, 210.0, False),
        (0.0, 0.0, False),
    ],
)
def test_dropped_allows_up_to_the_max_drop(base, head, fails):
    assert ev.dropped(base, head, 0.10) is fails


def test_codex_answer_keeps_only_agent_messages():
    events = "\n".join(
        [
            "Reading additional input from stdin...",
            '{"type": "thread.started", "thread_id": "t"}',
            '{"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}}',
            '{"type": "item.completed", "item": {"type": "agent_message", "text": "Use and_."}}',
            '{"type": "item.completed", "item": {"type": "agent_message", "text": "Done."}}',
        ]
    )
    assert ev.codex_answer(events) == "Use and_.\nDone."


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    templates = tmp_path / ev.TEMPLATES
    templates.mkdir(parents=True)
    (templates / "AGENTS.md").write_text(
        f"base\n{ev.EVALUATION_START}\neval\n{ev.EVALUATION_END}\n"
    )
    (templates / "CLAUDE.md").write_text("@AGENTS.md\n")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(
        tmp_path,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "base",
    )
    (templates / "AGENTS.md").write_text("head\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_write_project_takes_the_file_at_the_ref_or_the_working_tree(repo, tmp_path):
    ev.write_project(tmp_path / "base", "main")
    ev.write_project(tmp_path / "head", None)
    assert (tmp_path / "base" / "AGENTS.md").read_text() == "base\neval\n"
    assert (tmp_path / "head" / "AGENTS.md").read_text() == "head\n"
    assert (tmp_path / "head" / "CLAUDE.md").read_text() == "@AGENTS.md\n"


def test_write_project_rejects_an_unknown_ref(repo, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        ev.write_project(tmp_path / "base", "no-such-ref")


def _run(monkeypatch, answers: dict[str, str]):
    """Run the command with Claude replaced by canned answers keyed by project dir name."""
    monkeypatch.setattr(ev, "load_items", lambda: [ITEM, {**ITEM, "id": "X-002"}])
    monkeypatch.setattr(ev, "write_project", lambda dest, ref: dest.mkdir(parents=True))
    monkeypatch.setattr(
        ev, "ask_claude", lambda project, question: answers[project.name]
    )
    return CliRunner().invoke(
        ev.agents_md_eval, ["--agent", "claude", "--workers", "2"]
    )


def test_command_fails_when_the_branch_teaches_less(monkeypatch):
    result = _run(
        monkeypatch,
        {"base": "uv tool install crewai --upgrade", "head": "pip install -U crewai"},
    )
    assert result.exit_code == 1
    assert "dropped more than 10% below origin/main" in result.output
    assert "claude lost X-001" in result.output


def test_command_passes_when_the_branch_holds_or_improves(monkeypatch):
    result = _run(
        monkeypatch, {"base": "use uv tool", "head": "uv tool install crewai --upgrade"}
    )
    assert result.exit_code == 0, result.output
    assert "+100.0%" in result.output


def test_command_without_codex_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _run(monkeypatch, {"base": "x", "head": "x"})
    assert result.exit_code == 0, result.output


def test_codex_runs_in_its_own_existing_home_without_api_keys(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs, Path(kwargs["env"]["CODEX_HOME"]).is_dir()))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ask = ev.codex_asker(tmp_path / "codex-home")
    ask(tmp_path, "How?")

    (login, login_kwargs, home_existed), (_, exec_kwargs, _) = calls
    assert login[:2] == ["codex", "login"] and home_existed
    assert login_kwargs["input"] == "sk-test"
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} & set(exec_kwargs["env"])

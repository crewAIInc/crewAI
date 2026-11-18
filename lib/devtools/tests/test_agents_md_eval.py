from pathlib import Path
import re
import subprocess

from click.testing import CliRunner
from crewai_devtools import agents_md_eval as ev
import pytest


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
        (
            "`crewai update` is a different command; run `uv tool install crewai --upgrade`.",
            "right",
        ),
        (
            "`crewai update` does something different. Use `uv tool install crewai --upgrade`.",
            "right",
        ),
        (
            "Don\u2019t run `crewai update`; use `uv tool install crewai --upgrade`.",
            "right",
        ),
        (
            "`uv tool install crewai --upgrade` replaces `crewai update` for this.",
            "right",
        ),
        (
            "It doesn't need `crewai update`: run `uv tool install crewai --upgrade`.",
            "right",
        ),
        (
            "It doesn\u2019t need `crewai update`: run `uv tool install crewai --upgrade`.",
            "right",
        ),
        (
            "You can't use `crewai update`; run `uv tool install crewai --upgrade`.",
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
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)  # noqa: S603, S607 - fixed git command in a temp repo


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


def _run(monkeypatch, answers: dict[str, str], *args: str):
    """Run the command with Claude Code replaced by canned answers keyed by project dir."""
    items = [ITEM] + [{**ITEM, "id": f"X-{n:03d}"} for n in range(2, 101)]
    monkeypatch.setattr(ev, "load_items", lambda: items)
    monkeypatch.setattr(ev, "write_project", lambda dest, ref: dest.mkdir(parents=True))

    def fake(project, question):
        if not answers[project.name]:
            raise ev.NoAnswer("empty answer")
        return answers[project.name]

    monkeypatch.setattr(ev, "claude_asker", lambda model: fake)
    result = CliRunner().invoke(
        ev.agents_md_eval,
        ["--model", "claude:claude-haiku-5-5", "--workers", "4", *args],
    )
    # Rich wraps long lines at the runner's width; compare whitespace-normalized text.
    return result.exit_code, " ".join(result.output.split())


def test_command_fails_when_the_branch_teaches_less(monkeypatch):
    code, output = _run(
        monkeypatch,
        {"base": "uv tool install crewai --upgrade", "head": "pip install -U crewai"},
    )
    assert code == 1
    assert (
        "claude:claude-haiku-5-5: scored -100.0% against origin/main, past the 10% limit."
        in output
    )
    assert "claude:claude-haiku-5-5 lost X-001" in output


def test_command_passes_when_the_branch_holds_or_improves(monkeypatch):
    code, output = _run(
        monkeypatch, {"base": "use uv tool", "head": "uv tool install crewai --upgrade"}
    )
    assert code == 0, output
    assert "+100.0%" in output


def test_command_respects_a_custom_max_drop(monkeypatch):
    code, output = _run(
        monkeypatch,
        {"base": "uv tool install crewai --upgrade", "head": "use uv tool"},
        "--max-drop",
        "0.6",
    )
    assert code == 0, output
    assert "-50.0%" in output


def test_command_fails_when_sessions_give_no_answer(monkeypatch):
    code, output = _run(
        monkeypatch, {"base": "uv tool install crewai --upgrade", "head": ""}
    )
    assert code == 1
    assert "too many sessions gave no answer (0 / 100 of 100)" in output
    assert "gave no answer with the branch file: empty answer (100)" in output


def test_command_prints_brackets_in_items_and_reasons_as_text(monkeypatch):
    items = [
        {**ITEM, "id": f"X-{n:03d}", "item": "Set `[tool.crewai]` in pyproject.toml."}
        for n in range(1, 101)
    ]
    monkeypatch.setattr(ev, "load_items", lambda: items)
    monkeypatch.setattr(ev, "write_project", lambda dest, ref: dest.mkdir(parents=True))

    def fake(project, question):
        if project.name == "head":
            raise ev.NoAnswer("exit 1: [Errno 8] nodename nor servname provided")
        return "uv tool install crewai --upgrade"

    monkeypatch.setattr(ev, "claude_asker", lambda model: fake)
    result = CliRunner().invoke(
        ev.agents_md_eval, ["--model", "claude:claude-haiku-5-5"]
    )
    output = " ".join(result.output.split())

    assert result.exit_code == 1
    assert "[Errno 8] nodename nor servname provided (100)" in output
    assert "lost X-001: Set `[tool.crewai]` in pyproject.toml." in output


def test_command_reads_the_head_file_at_the_ref_it_is_given(monkeypatch):
    refs: dict[str, str | None] = {}

    def write_project(dest, ref):
        refs[dest.name] = ref
        dest.mkdir(parents=True)

    monkeypatch.setattr(ev, "write_project", write_project)
    monkeypatch.setattr(ev, "load_items", lambda: [ITEM])
    monkeypatch.setattr(
        ev, "claude_asker", lambda model: lambda project, question: "uv tool"
    )
    CliRunner().invoke(
        ev.agents_md_eval,
        ["--model", "claude:claude-haiku-5-5", "--base", "abc123", "--head", "def456"],
    )

    assert refs == {"base": "abc123", "head": "def456"}


def test_command_without_codex_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code, output = _run(monkeypatch, {"base": "x", "head": "x"})
    assert code == 0, output


@pytest.mark.parametrize(
    "spec", ["claude", "codex:", "gemini:gemini-3", ":gpt-5.6-sol"]
)
def test_command_rejects_a_malformed_model(spec):
    result = CliRunner().invoke(ev.agents_md_eval, ["--model", spec])
    assert result.exit_code == 2
    assert "use claude:<model> or codex:<model>" in result.output


def test_default_models_are_the_four_the_action_runs():
    assert [ev.parse_model(spec) for spec in ev.MODELS] == [
        ("claude", "claude-haiku-5-5"),
        ("claude", "claude-sonnet-5-5"),
        ("codex", "gpt-5.6-luna"),
        ("codex", "gpt-5.6-sol"),
    ]


def test_answer_retries_once_then_reports_why_there_was_none():
    replies = iter([subprocess.TimeoutExpired("claude", 300), "Use and_."])

    def flaky(project, question):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def silent(project, question):
        raise ev.NoAnswer("exit 1: overloaded")

    assert ev.answer(flaky, Path("."), "How?") == ("Use and_.", "")
    assert ev.answer(silent, Path("."), "How?") == (None, "exit 1: overloaded")


def test_unanswered_sessions_score_zero_and_are_counted_by_reason(tmp_path):
    def half(project, question):
        if question == "second":
            raise ev.NoAnswer("empty answer")
        return "uv tool install crewai --upgrade"

    items = [{**ITEM, "id": "A", "q": "first"}, {**ITEM, "id": "B", "q": "second"}]
    totals, missing = ev.score(half, tmp_path, items, reps=2, workers=2)
    assert totals == {"A": 1.0, "B": 0.0}
    assert missing == {"empty answer": 2}


def test_codex_runs_in_its_own_existing_home_without_api_keys(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs, Path(kwargs["env"]["CODEX_HOME"]).is_dir()))
        message = '{"type": "item.completed", "item": {"type": "agent_message", "text": "Use and_."}}'
        return subprocess.CompletedProcess(argv, 0, stdout=message, stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ask = ev.codex_asker(tmp_path / "codex-home", "gpt-5.6-sol")
    assert ask(tmp_path, "How?") == "Use and_."

    (login, login_kwargs, home_existed), (exec_argv, exec_kwargs, _) = calls
    assert login[:2] == ["codex", "login"] and home_existed
    assert login_kwargs["input"] == "sk-test"
    assert exec_argv[exec_argv.index("-m") + 1] == "gpt-5.6-sol"
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} & set(exec_kwargs["env"])


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "reason"),
    [
        (
            1,
            "partial",
            "Reading input...\nAPI Error: 529 overloaded\n",
            "exit 1: API Error: 529 overloaded",
        ),
        (2, "", "", "exit 2: no error output"),
        (
            1,
            "Not logged in · Please run /login\n",
            "",
            "exit 1: Not logged in · Please run /login",
        ),
        (0, "  ", "", "empty answer"),
        (1, "", "Invalid API key sk-ant-abc123XYZ\n", "exit 1: Invalid API key sk-***"),
        (
            1,
            "Invalid API key · Fix external API key\n",
            '{"model":"claude-haiku-5-5","query_source":"sdk"}\n',
            "exit 1: Invalid API key · Fix external API key",
        ),
    ],
)
def test_a_session_without_an_answer_says_why(
    tmp_path, monkeypatch, returncode, stdout, stderr, reason
):
    monkeypatch.setattr(
        ev.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, returncode, stdout=stdout, stderr=stderr
        ),
    )
    with pytest.raises(ev.NoAnswer, match=re.escape(reason)):
        ev.claude_asker("claude-haiku-5-5")(tmp_path, "How?")


EVAL_WORKFLOW = ".github/workflows/agents-md-eval.yml"


def _eval_workflow():
    import yaml

    return yaml.safe_load(ev.read_at(None, EVAL_WORKFLOW))


def test_the_workflow_runs_the_default_models():
    job = _eval_workflow()["jobs"]["eval"]
    assert job["strategy"]["matrix"]["model"] == list(ev.MODELS)


def test_the_eval_runs_on_pull_requests_against_main_from_this_repository_only():
    workflow = _eval_workflow()
    triggers = workflow[True]  # PyYAML reads the key `on` as the boolean true
    assert list(triggers) == ["pull_request_target"]
    assert triggers["pull_request_target"]["branches"] == ["main"]
    assert triggers["pull_request_target"]["paths"] == [
        "lib/cli/src/crewai_cli/templates/AGENTS.md",
        "lib/cli/src/crewai_cli/templates/CLAUDE.md",
    ]
    assert (
        workflow["jobs"]["eval"]["if"]
        == "github.event.pull_request.head.repo.full_name == github.repository"
    )


def test_the_eval_runs_only_the_base_branchs_code_with_the_keys():
    """The keys reach only code from the base commit; the PR's commit is fetched as data."""
    job = _eval_workflow()["jobs"]["eval"]
    steps = job["steps"]
    checkout = next(
        step for step in steps if "actions/checkout" in step.get("uses", "")
    )
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    run = next(
        step for step in steps if "devtools agents-md-eval" in step.get("run", "")
    )
    assert '--base "$BASE_SHA" --head "$HEAD_SHA"' in run["run"]
    assert run["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert job["environment"]["name"] == "agents-md-eval"


def test_the_eval_fetches_the_pull_request_only_when_it_runs():
    steps = _eval_workflow()["jobs"]["eval"]["steps"]
    names = [step["name"] for step in steps]
    fetch = next(step for step in steps if "git fetch" in step.get("run", ""))
    assert names.index(fetch["name"]) > names.index("Check the eval and the API keys")
    assert fetch["if"] == "steps.keys.outputs.run == 'true'"

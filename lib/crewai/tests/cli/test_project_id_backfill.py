"""Every user-invoked CLI command that touches a project backfills its `project_id`.

`crewai run` has always backfilled: a project that declares `[tool.crewai]` but no
`project_id` gets one minted the first time the user runs it. No other command did,
so a project driven entirely through `crewai test`, `crewai deploy` or
`crewai traces enable` never acquired an id and every one of its runs stayed
unattributable.

These commands are all actions the user explicitly invoked, which is the same
condition `run_crew` already relies on, so extending the backfill needs no new
policy. It is still never called from the SDK during kickoff, and
`get_or_create_project_id` still refuses to create the `[tool.crewai]` table, so an
unrelated directory is never rewritten.

The patched backfill RAISES here rather than returning. That proves the call happened
and simultaneously guarantees nothing after it runs, so no test touches real user
settings, spawns a subprocess, or reaches the network.
"""

from pathlib import Path
from unittest import mock

from click.testing import CliRunner
from crewai_cli.cli import crewai
import pytest


class _BackfillReached(Exception):
    """Raised by the patched backfill so the command stops at that point."""


# (test id, argv). Args are the minimum click accepts; the command body is never
# reached beyond the backfill call, so nothing here needs to be a valid target.
COMMANDS = [
    ("train", ["train", "-n", "1"]),
    ("replay", ["replay", "-t", "task-1"]),
    ("test", ["test"]),
    ("login", ["login"]),
    ("deploy_create", ["deploy", "create"]),
    ("deploy_push", ["deploy", "push"]),
    ("flow_add_crew", ["flow", "add-crew", "some_crew"]),
    ("enterprise_configure", ["enterprise", "configure", "https://example.test"]),
    ("traces_enable", ["traces", "enable"]),
]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(("name", "argv"), COMMANDS, ids=[c[0] for c in COMMANDS])
def test_command_backfills_project_id(runner, name, argv):
    with mock.patch(
        "crewai_cli.cli.get_or_create_project_id",
        side_effect=_BackfillReached,
    ) as backfill:
        result = runner.invoke(crewai, argv)

    assert backfill.called, (
        f"`crewai {' '.join(argv)}` did not backfill project_id, so a project driven "
        f"only through this command never becomes attributable"
    )
    assert isinstance(result.exception, _BackfillReached), (
        "the backfill must run before the command does any other work, so that a "
        "command which later fails still leaves the project with an id"
    )


MINIMAL_PYPROJECT = """\
[project]
name = "demo"
version = "0.1.0"

[tool.crewai]
"""


def _in_a_project(runner):
    """A cwd that `crewai run` accepts, so execution reaches the backfill call."""
    return runner.isolated_filesystem()


def test_run_still_backfills(runner):
    """`crewai run` was already correct and must stay that way.

    Asserted at runtime rather than by reading the source, so a reformat cannot
    break it and a real regression cannot slip past it.
    """
    with _in_a_project(runner):
        Path("pyproject.toml").write_text(MINIMAL_PYPROJECT, encoding="utf-8")
        with mock.patch(
            "crewai_cli.run_crew.get_or_create_project_id",
            side_effect=_BackfillReached,
        ) as backfill:
            result = runner.invoke(crewai, ["run"])

    assert backfill.called, "`crewai run` stopped backfilling project_id"
    assert isinstance(result.exception, _BackfillReached)


def test_flow_kickoff_delegates_the_backfill_and_does_not_duplicate_it(runner):
    """`crewai flow kickoff` must inherit the backfill from run_crew, not repeat it.

    A second call would mint under a lock run_crew is about to take -- wasted work
    rather than a correctness bug, but it would also hide the delegation from anyone
    reading the command. Asserted by call counts on the two distinct import sites,
    which is what makes "delegates" and "duplicates" distinguishable at runtime.
    """
    with _in_a_project(runner):
        Path("pyproject.toml").write_text(MINIMAL_PYPROJECT, encoding="utf-8")
        with (
            mock.patch("crewai_cli.cli.get_or_create_project_id") as in_cli,
            mock.patch(
                "crewai_cli.run_crew.get_or_create_project_id",
                side_effect=_BackfillReached,
            ) as in_run_crew,
        ):
            runner.invoke(crewai, ["flow", "kickoff"])

    assert in_run_crew.call_count == 1, (
        "flow kickoff must reach run_crew's backfill exactly once"
    )
    assert not in_cli.called, (
        "flow kickoff must not add its own backfill call; it delegates to run_crew"
    )

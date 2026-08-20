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


def test_run_still_backfills():
    """`crewai run` was already correct and must stay that way.

    Asserted at the source rather than by invoking it, because `run_crew` reads the
    cwd and this only needs to pin that the call site still exists.
    """
    from pathlib import Path

    import crewai_cli.run_crew as run_crew_module

    source = Path(run_crew_module.__file__).read_text(encoding="utf-8")
    assert "get_or_create_project_id()" in source


def test_flow_kickoff_inherits_the_backfill_from_run():
    """`crewai flow kickoff` delegates to run_crew, so it must NOT call it twice.

    Pinned deliberately: adding a second call here would mint under a lock that
    run_crew is about to take, which is wasted work rather than a correctness bug -
    but it would also hide the delegation from anyone reading the command.
    """
    from pathlib import Path

    import crewai_cli.cli as cli_module

    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    kickoff = source.split('@flow.command(name="kickoff")')[1].split("@flow.command")[0]
    assert "run_crew(" in kickoff
    assert "get_or_create_project_id()" not in kickoff

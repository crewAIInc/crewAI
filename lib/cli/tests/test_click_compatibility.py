"""Tests for click dependency compatibility.

Regression tests for https://github.com/crewAIInc/crewAI/issues/6002
The click dependency was previously pinned to ~=8.1.7 (i.e. >=8.1.7,<8.2.0)
which prevented users from upgrading to click 8.2+ as required by their
security policies. The constraint has been widened to >=8.1.7,<9 to allow
newer click 8.x releases while still guarding against a future major version
break.
"""

from importlib.metadata import requires
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from packaging.requirements import Requirement


# ---------------------------------------------------------------------------
# Verify the runtime click version satisfies the declared constraint
# ---------------------------------------------------------------------------

def _get_click_requirement_from_pyproject(package_dir: str) -> Requirement:
    """Parse the click requirement directly from a pyproject.toml file."""
    import tomli

    pyproject_path = Path(__file__).resolve().parents[3] / package_dir / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)
    deps = data["project"]["dependencies"]
    for dep in deps:
        req = Requirement(dep)
        if req.name == "click":
            return req
    raise ValueError(f"click not found in {pyproject_path}")


@pytest.mark.parametrize(
    "package_dir",
    [
        "lib/crewai",
        "lib/cli",
        "lib/devtools",
    ],
)
def test_click_constraint_allows_8_3_3(package_dir: str):
    """The declared click constraint must accept click 8.3.3 (issue #6002)."""
    req = _get_click_requirement_from_pyproject(package_dir)
    # packaging's Requirement.specifier supports `__contains__` for version checks
    assert "8.3.3" in req.specifier, (
        f"{package_dir}: click constraint {req.specifier} does not allow 8.3.3"
    )


@pytest.mark.parametrize(
    "package_dir",
    [
        "lib/crewai",
        "lib/cli",
        "lib/devtools",
    ],
)
def test_click_constraint_allows_8_1_7(package_dir: str):
    """The declared click constraint must still accept the original minimum (8.1.7)."""
    req = _get_click_requirement_from_pyproject(package_dir)
    assert "8.1.7" in req.specifier, (
        f"{package_dir}: click constraint {req.specifier} does not allow 8.1.7"
    )


@pytest.mark.parametrize(
    "package_dir",
    [
        "lib/crewai",
        "lib/cli",
        "lib/devtools",
    ],
)
def test_click_constraint_rejects_next_major(package_dir: str):
    """The declared click constraint must reject click 9.0.0."""
    req = _get_click_requirement_from_pyproject(package_dir)
    assert "9.0.0" not in req.specifier, (
        f"{package_dir}: click constraint {req.specifier} should not allow 9.0.0"
    )


# ---------------------------------------------------------------------------
# Verify the installed click version works with the CLI
# ---------------------------------------------------------------------------

def test_click_version_is_compatible():
    """The installed click version must be within the 8.x range."""
    major = int(click.__version__.split(".")[0])
    assert major == 8, f"Expected click 8.x, got {click.__version__}"


def test_cli_runner_works_with_installed_click():
    """Smoke-test: CliRunner from the installed click can invoke a trivial command."""

    @click.command()
    @click.option("--name", default="world")
    def hello(name: str) -> None:
        click.echo(f"Hello {name}!")

    runner = CliRunner()
    result = runner.invoke(hello, ["--name", "crewai"])
    assert result.exit_code == 0
    assert "Hello crewai!" in result.output


def test_cli_group_works_with_installed_click():
    """Smoke-test: click.group, click.option, click.argument all work."""

    @click.group()
    def grp() -> None:
        pass

    @grp.command()
    @click.argument("task")
    @click.option("--verbose", is_flag=True)
    def run(task: str, verbose: bool) -> None:
        if verbose:
            click.echo(f"Running {task} (verbose)")
        else:
            click.echo(f"Running {task}")

    runner = CliRunner()
    result = runner.invoke(grp, ["run", "test-task", "--verbose"])
    assert result.exit_code == 0
    assert "Running test-task (verbose)" in result.output

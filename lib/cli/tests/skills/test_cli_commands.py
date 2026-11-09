"""Tests for the top-level `crewai skill` command group.

The Skills Repository graduated from the gated `crewai experimental skill`
group: the commands must be reachable at `crewai skill ...` without
CREWAI_EXPERIMENTAL set.
"""

from unittest import mock

from click.testing import CliRunner
from crewai_cli.cli import crewai
import pytest


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def no_experimental_env(monkeypatch):
    monkeypatch.delenv("CREWAI_EXPERIMENTAL", raising=False)


class TestSkillGroupIsTopLevel:
    def test_skill_group_registered_at_top_level(self, runner, no_experimental_env):
        result = runner.invoke(crewai, ["skill", "--help"])
        assert result.exit_code == 0, result.output
        for subcommand in ("create", "install", "publish", "list", "search"):
            assert subcommand in result.output

    def test_skill_group_not_gated_by_experimental_env(
        self, runner, no_experimental_env
    ):
        with mock.patch("crewai_cli.skills.main.SkillCommand") as mock_cmd:
            result = runner.invoke(crewai, ["skill", "create", "my-skill"])
        assert result.exit_code == 0, result.output
        mock_cmd.return_value.create.assert_called_once_with(
            "my-skill", in_project=True
        )

    def test_skill_create_no_project_flag(self, runner, no_experimental_env):
        with mock.patch("crewai_cli.skills.main.SkillCommand") as mock_cmd:
            result = runner.invoke(crewai, ["skill", "create", "my-skill", "--no-project"])
        assert result.exit_code == 0, result.output
        mock_cmd.return_value.create.assert_called_once_with(
            "my-skill", in_project=False
        )


class TestSkillPublishIsOrgScopedOnly:
    def test_publish_rejects_public_flag(self, runner, no_experimental_env):
        result = runner.invoke(crewai, ["skill", "publish", "--public"])
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_publish_passes_org_and_force_only(self, runner, no_experimental_env):
        with mock.patch("crewai_cli.skills.main.SkillCommand") as mock_cmd:
            result = runner.invoke(
                crewai, ["skill", "publish", "--org", "acme", "--force"]
            )
        assert result.exit_code == 0, result.output
        mock_cmd.return_value.publish.assert_called_once_with(org="acme", force=True)


class TestSkillSearchCommand:
    @pytest.mark.parametrize(
        ("args", "query"), [(["support triage"], "support triage"), ([], None)]
    )
    def test_search_passes_the_optional_query(
        self, runner, no_experimental_env, args, query
    ):
        with mock.patch("crewai_cli.skills.main.SkillCommand") as mock_cmd:
            result = runner.invoke(crewai, ["skill", "search", *args])
        assert result.exit_code == 0, result.output
        mock_cmd.return_value.search.assert_called_once_with(query)

    def test_list_still_shows_installed_skills_only(self, runner, no_experimental_env):
        with mock.patch("crewai_cli.skills.main.SkillCommand") as mock_cmd:
            result = runner.invoke(crewai, ["skill", "list"])
        assert result.exit_code == 0, result.output
        mock_cmd.return_value.list_cached.assert_called_once_with()
        mock_cmd.return_value.search.assert_not_called()

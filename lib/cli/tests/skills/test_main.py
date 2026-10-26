"""Tests for SkillCommand CLI."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from crewai_cli.shared.token_manager import TokenManager


if TYPE_CHECKING:
    from crewai_cli.skills.main import SkillCommand


@contextmanager
def in_temp_dir():
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            yield td
        finally:
            os.chdir(original)


@pytest.fixture
def skill_command():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(
            TokenManager, "_get_secure_storage_path", return_value=Path(temp_dir)
        ):
            TokenManager().save_tokens(
                "test-token", (datetime.now() + timedelta(seconds=36000)).timestamp()
            )
            from crewai_cli.skills.main import SkillCommand
            cmd = SkillCommand()
            yield cmd


class TestSkillCreate:
    def test_create_in_project(self, skill_command, tmp_path):
        with in_temp_dir():
            Path("pyproject.toml").write_text("[tool.poetry]\nname = 'test'\n")
            skill_command.create("my-skill")
            assert Path("skills/my-skill/SKILL.md").exists()
            assert Path("skills/my-skill/scripts").is_dir()
            assert Path("skills/my-skill/references").is_dir()
            assert Path("skills/my-skill/assets").is_dir()

    def test_create_outside_project(self, skill_command, tmp_path):
        with in_temp_dir():
            skill_command.create("standalone-skill", in_project=False)
            assert Path("standalone-skill/SKILL.md").exists()

    def test_create_adds_name_to_skill_md(self, skill_command):
        with in_temp_dir():
            skill_command.create("hello-world", in_project=False)
            content = Path("hello-world/SKILL.md").read_text()
            assert "name: hello-world" in content
            assert "version: 0.1.0" in content

    def test_create_fails_if_dir_exists(self, skill_command):
        with in_temp_dir():
            Path("existing-skill").mkdir()
            with pytest.raises(SystemExit):
                skill_command.create("existing-skill", in_project=False)


class TestSkillInstall:
    def _zip_skill(self, name: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("SKILL.md", f"---\nname: {name}\ndescription: Test.\n---\nInstructions.")
        return buf.getvalue()

    def test_install_invalid_ref_no_at(self, skill_command):
        with pytest.raises(SystemExit):
            skill_command.install("acme/my-skill")

    def test_install_invalid_ref_no_slash(self, skill_command):
        with pytest.raises(SystemExit):
            skill_command.install("@acmeskill")

    def test_install_404(self, skill_command):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        skill_command.plus_api_client.get_skill = MagicMock(return_value=mock_resp)

        with pytest.raises(SystemExit):
            skill_command.install("@acme/ghost")

    def test_install_in_project(self, skill_command):
        import base64
        archive = self._zip_skill("my-skill")
        encoded = "data:application/zip;base64," + base64.b64encode(archive).decode()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"file": encoded, "version": "1.0.0"}
        skill_command.plus_api_client.get_skill = MagicMock(return_value=mock_resp)

        with in_temp_dir():
            Path("pyproject.toml").write_text("[tool]\n")
            skill_command.install("@acme/my-skill")
            assert Path("skills/my-skill/SKILL.md").exists()


class TestSkillPublish:
    def test_publish_no_skill_md(self, skill_command):
        with in_temp_dir():
            with pytest.raises(SystemExit):
                skill_command.publish(org="acme")

    def test_publish_missing_version(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: Test.\n---\nInstructions."
            )
            with pytest.raises(SystemExit):
                skill_command.publish(org="acme")

    def test_publish_missing_name(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\ndescription: Test.\nversion: 1.0.0\n---\nInstructions."
            )
            with pytest.raises(SystemExit):
                skill_command.publish(org="acme")

    def test_publish_no_org(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\nversion: 1.0.0\ndescription: Test.\n---\nInstructions."
            )
            with patch.object(skill_command, "plus_api_client") as mock_client:
                mock_resp = MagicMock()
                mock_resp.is_success = True
                mock_resp.status_code = 200
                mock_resp.json.return_value = {}
                mock_client.publish_skill.return_value = mock_resp
                with patch("crewai_cli.skills.main.Settings") as mock_settings_cls:
                    mock_settings_cls.return_value.org_name = None
                    mock_settings_cls.return_value.enterprise_base_url = None
                    with pytest.raises(SystemExit):
                        skill_command.publish(org=None)

    def test_publish_calls_api(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            skill_command.plus_api_client.publish_skill = MagicMock(return_value=mock_resp)
            with patch("crewai_cli.skills.main.Settings") as mock_settings_cls:
                mock_settings_cls.return_value.org_name = "acme"
                mock_settings_cls.return_value.enterprise_base_url = None

                skill_command.publish(org="acme")

            skill_command.plus_api_client.publish_skill.assert_called_once()
            call_kwargs = skill_command.plus_api_client.publish_skill.call_args
            assert call_kwargs.kwargs["name"] == "my-skill"
            assert call_kwargs.kwargs["version"] == "1.0.0"
            # Skills are always org-scoped; there is no public visibility.
            assert call_kwargs.kwargs["is_public"] is False

    def test_publish_blocked_when_git_not_synced(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            skill_command.plus_api_client.publish_skill = MagicMock()
            with patch("crewai_cli.skills.main.git.Repository") as mock_repo_cls:
                mock_repo_cls.return_value.is_synced.return_value = False
                with pytest.raises(SystemExit):
                    skill_command.publish(org="acme")
            skill_command.plus_api_client.publish_skill.assert_not_called()

    def test_publish_force_skips_git_check(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            skill_command.plus_api_client.publish_skill = MagicMock(return_value=mock_resp)
            with (
                patch("crewai_cli.skills.main.git.Repository") as mock_repo_cls,
                patch("crewai_cli.skills.main.Settings") as mock_settings_cls,
            ):
                mock_settings_cls.return_value.org_name = "acme"
                mock_settings_cls.return_value.enterprise_base_url = None
                skill_command.publish(org="acme", force=True)
            mock_repo_cls.assert_not_called()
            skill_command.plus_api_client.publish_skill.assert_called_once()

    def test_publish_blocked_when_fetch_fails(self, skill_command):
        """A failing remote fetch must fail closed, not silently skip syncing."""
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            skill_command.plus_api_client.publish_skill = MagicMock()
            with patch("crewai_cli.skills.main.git.Repository") as mock_repo_cls:
                mock_repo_cls.return_value.fetch.side_effect = ValueError(
                    "Git fetch failed with exit code 128"
                )
                with pytest.raises(SystemExit):
                    skill_command.publish(org="acme")
            skill_command.plus_api_client.publish_skill.assert_not_called()

    def test_publish_blocked_when_git_state_cannot_be_validated(self, skill_command):
        """Git errors other than 'not a repo' must fail closed, not skip the check."""
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            skill_command.plus_api_client.publish_skill = MagicMock()
            with patch(
                "crewai_cli.skills.main.git.Repository",
                side_effect=ValueError("Git fetch failed with exit code 128"),
            ):
                with pytest.raises(SystemExit):
                    skill_command.publish(org="acme")
            skill_command.plus_api_client.publish_skill.assert_not_called()

    def test_publish_proceeds_outside_git_repo(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: A test skill.\nmetadata:\n  version: 1.0.0\n---\nInstructions."
            )
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            skill_command.plus_api_client.publish_skill = MagicMock(return_value=mock_resp)
            with (
                patch(
                    "crewai_cli.skills.main.git.Repository",
                    side_effect=ValueError("not a Git repository"),
                ),
                patch("crewai_cli.skills.main.Settings") as mock_settings_cls,
            ):
                mock_settings_cls.return_value.org_name = "acme"
                mock_settings_cls.return_value.enterprise_base_url = None
                skill_command.publish(org="acme")
            skill_command.plus_api_client.publish_skill.assert_called_once()


class TestSkillListCached:
    @pytest.mark.parametrize("metadata", [b"\xff\xfe", b"{invalid json"])
    def test_list_cached_skips_unreadable_metadata(
        self,
        skill_command: SkillCommand,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        metadata: bytes,
    ) -> None:
        cache_root = tmp_path / ".crewai" / "skills" / "acme"
        broken = cache_root / "a-broken"
        healthy = cache_root / "z-healthy"
        broken.mkdir(parents=True)
        healthy.mkdir()
        (broken / ".crewai_meta.json").write_bytes(metadata)
        (healthy / ".crewai_meta.json").write_text(
            json.dumps({"org": "acme", "name": "z-healthy", "version": "1.0.0"}),
            encoding="utf-8",
        )

        with (
            in_temp_dir(),
            patch("crewai_cli.skills.main.Path.home", return_value=tmp_path),
        ):
            skill_command.list_cached()

        output = capsys.readouterr().out
        assert "@acme/z-healthy" in output
        assert "skipping malformed cache entry" in output
        assert (broken / ".crewai_meta.json").read_bytes() == metadata

    def test_list_cached_empty(self, skill_command, capsys):
        with in_temp_dir():
            skill_command.list_cached()

    def test_list_cached_shows_project_skills(self, skill_command, capsys):
        with in_temp_dir():
            skill_dir = Path("skills/my-skill")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\nversion: 0.5.0\ndescription: A skill.\n---\nBody."
            )
            skill_command.list_cached()


ORG_UUID = "6f1c2a9e-3b7d-4c11-9a2e-8b0f4d2c1e77"


def _skill_row(
    handle: str,
    *,
    description: str | None = "A skill.",
    latest_version: str | None = "1.0.0",
    scan: str | None = "passed",
) -> dict[str, object]:
    return {
        "handle": handle,
        "description": description,
        "license": None,
        "is_public": False,
        "latest_version": latest_version,
        "virus_scanning_status": scan,
        "created_at": "2026-10-01T00:00:00Z",
    }


def _skills_page(rows: list[dict[str, object]], total_pages: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": rows,
            "meta": {
                "current_page": 1,
                "total_pages": total_pages,
                "total_count": len(rows),
                "per_page": 25,
            },
        },
    )


class TestSkillSearch:
    def _search(
        self,
        skill_command: SkillCommand,
        query: str | None,
        responses: list[httpx.Response],
        org_uuid: str | None = ORG_UUID,
    ) -> MagicMock:
        list_skills = MagicMock(side_effect=responses)
        skill_command.plus_api_client.list_skills = list_skills
        with patch("crewai_cli.skills.main.Settings") as mock_settings_cls:
            mock_settings_cls.return_value.org_uuid = org_uuid
            mock_settings_cls.return_value.org_name = "Acme" if org_uuid else None
            skill_command.search(query)
        return list_skills

    def test_lists_every_page_with_the_query(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        list_skills = self._search(
            skill_command,
            "triage",
            [
                _skills_page([_skill_row("ticket-triage")], total_pages=2),
                _skills_page([_skill_row("triage-rules")], total_pages=2),
            ],
        )

        output = capsys.readouterr().out
        assert "ticket-triage@1.0.0" in output
        assert "triage-rules@1.0.0" in output
        assert list_skills.call_args_list == [
            call(search="triage", page=1),
            call(search="triage", page=2),
        ]

    def test_prints_the_ref_form_for_the_current_organization(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._search(skill_command, None, [_skills_page([_skill_row("brand-voice")], 1)])

        output = capsys.readouterr().out
        assert f'Attach as "@{ORG_UUID}/<handle>@<version>":' in output
        assert "brand-voice@1.0.0  scan: passed  A skill." in output.splitlines()

    def test_long_handles_and_descriptions_are_never_cut_when_piped(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # capsys is not a terminal, so rich renders at 80 columns, as it does
        # for a coding agent reading piped output.
        handle = "customer-support-escalation-triage-playbook-for-enterprise-tier"
        description = "Routes escalations by severity and account tier. " * 4
        self._search(
            skill_command,
            None,
            [_skills_page([_skill_row(handle, description=description)], 1)],
        )

        output = capsys.readouterr().out
        assert "…" not in output
        assert (
            f"{handle}@1.0.0  scan: passed  {description.strip()}"
            in output.splitlines()
        )

    def test_collapses_multiline_descriptions_and_prints_markup_literally(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        row = _skill_row(
            "review-rules", description="Checks [bold]PRs[/bold].\n\nTwice.\n"
        )
        self._search(skill_command, None, [_skills_page([row], 1)])

        assert (
            "review-rules@1.0.0  scan: passed  Checks [bold]PRs[/bold]. Twice."
            in capsys.readouterr().out.splitlines()
        )

    def test_shows_a_skill_without_version_or_description(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        row = _skill_row("draft", description=None, latest_version=None, scan=None)
        self._search(skill_command, None, [_skills_page([row], 1)])

        assert "draft  scan: -" in capsys.readouterr().out.splitlines()

    def test_without_an_organization_points_to_org_switch(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._search(
            skill_command, None, [_skills_page([_skill_row("brand-voice")], 1)], None
        )

        output = capsys.readouterr().out
        assert "`crewai org switch <org_id>`" in " ".join(output.split())
        assert "Attach as" not in output
        assert "brand-voice@1.0.0" in output

    @pytest.mark.parametrize(
        ("query", "message"),
        [("nothing", "No skills match 'nothing'."), (None, "No skills found.")],
    )
    def test_empty_result_says_so(
        self,
        skill_command: SkillCommand,
        capsys: pytest.CaptureFixture[str],
        query: str | None,
        message: str,
    ) -> None:
        list_skills = self._search(skill_command, query, [_skills_page([], 0)])

        assert message in capsys.readouterr().out
        list_skills.assert_called_once_with(search=query, page=1)

    def test_unauthorized_exits_1_and_points_to_login(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            self._search(
                skill_command,
                None,
                [httpx.Response(401, json={"message": "Unauthorized"})],
            )

        assert exc_info.value.code == 1
        assert "Not authorized to list skills. Run `crewai login`." in (
            capsys.readouterr().out
        )

    def test_failure_on_a_later_page_exits_1_with_amp_message(
        self, skill_command: SkillCommand, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            self._search(
                skill_command,
                None,
                [
                    _skills_page([_skill_row("brand-voice")], total_pages=2),
                    httpx.Response(
                        500,
                        json={
                            "error": "internal_server_error",
                            "message": "database is down",
                        },
                    ),
                ],
            )

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Failed to list skills: 500" in output
        assert "database is down" in output
        assert "brand-voice@1.0.0" not in output

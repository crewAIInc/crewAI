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
from unittest.mock import MagicMock, patch

import pytest

from crewai_cli.shared.token_manager import TokenManager


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


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestSkillCreate:
    def test_create_in_project(self, skill_command, tmp_path):
        with in_temp_dir():
            # Simulate being inside a project
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


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------

class TestSkillPublish:
    def test_publish_no_skill_md(self, skill_command):
        with in_temp_dir():
            with pytest.raises(SystemExit):
                skill_command.publish(is_public=True, org="acme")

    def test_publish_missing_version(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\ndescription: Test.\n---\nInstructions."
            )
            with pytest.raises(SystemExit):
                skill_command.publish(is_public=True, org="acme")

    def test_publish_missing_name(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\ndescription: Test.\nversion: 1.0.0\n---\nInstructions."
            )
            with pytest.raises(SystemExit):
                skill_command.publish(is_public=True, org="acme")

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
                # No org set → should SystemExit (no org_name in settings)
                with patch("crewai_cli.skills.main.Settings") as mock_settings_cls:
                    mock_settings_cls.return_value.org_name = None
                    mock_settings_cls.return_value.enterprise_base_url = None
                    with pytest.raises(SystemExit):
                        skill_command.publish(is_public=True, org=None)

    def test_publish_calls_api(self, skill_command):
        with in_temp_dir():
            Path("SKILL.md").write_text(
                "---\nname: my-skill\nversion: 1.0.0\ndescription: A test skill.\n---\nInstructions."
            )
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.status_code = 200
            mock_resp.json.return_value = {}
            skill_command.plus_api_client.publish_skill = MagicMock(return_value=mock_resp)
            with patch("crewai_cli.skills.main.Settings") as mock_settings_cls:
                mock_settings_cls.return_value.org_name = "acme"
                mock_settings_cls.return_value.enterprise_base_url = None

                skill_command.publish(is_public=False, org="acme")

            skill_command.plus_api_client.publish_skill.assert_called_once()
            call_kwargs = skill_command.plus_api_client.publish_skill.call_args
            assert call_kwargs.kwargs["name"] == "my-skill"
            assert call_kwargs.kwargs["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# list_cached
# ---------------------------------------------------------------------------

class TestSkillListCached:
    def test_list_cached_empty(self, skill_command, capsys):
        with in_temp_dir():
            skill_command.list_cached()
            # Should not raise

    def test_list_cached_shows_project_skills(self, skill_command, capsys):
        with in_temp_dir():
            skill_dir = Path("skills/my-skill")
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\nversion: 0.5.0\ndescription: A skill.\n---\nBody."
            )
            skill_command.list_cached()
            # Should complete without error

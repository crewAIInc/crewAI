from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from crewai.context import platform_context
from crewai.skills.cache import SkillCacheManager
from crewai.skills.registry import (
    download_skill,
    is_registry_ref,
    parse_registry_ref,
)
import pytest


def _skill_archive(name: str) -> bytes:
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "SKILL.md",
            f"---\nname: {name}\ndescription: Test skill.\n---\n\nInstructions.",
        )
    return archive.getvalue()


def _mock_skill_response(name: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "latest_version": "1.0.0",
        "file": base64.b64encode(_skill_archive(name)).decode(),
    }
    return response


class TestIsRegistryRef:
    def test_at_prefixed(self) -> None:
        assert is_registry_ref("@acme/my-skill") is True

    def test_plain_string(self) -> None:
        assert is_registry_ref("my-skill") is False

    def test_path_like_string(self) -> None:
        assert is_registry_ref("./skills/my-skill") is False

    def test_non_string(self) -> None:
        assert is_registry_ref(None) is False
        assert is_registry_ref(42) is False
        assert is_registry_ref(Path("something")) is False


class TestParseRegistryRef:
    def test_valid(self) -> None:
        assert parse_registry_ref("@acme/my-skill") == ("acme", "my-skill")

    def test_valid_with_dashes(self) -> None:
        assert parse_registry_ref("@my-org/cool-skill") == ("my-org", "cool-skill")

    def test_missing_at(self) -> None:
        with pytest.raises(ValueError, match="must start with '@'"):
            parse_registry_ref("acme/my-skill")

    def test_missing_slash(self) -> None:
        with pytest.raises(ValueError, match="'@org/name' format"):
            parse_registry_ref("@acme-skill")

    def test_empty_org(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_registry_ref("@/my-skill")

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            parse_registry_ref("@acme/")


class TestResolveRegistryRef:
    def _make_skill_dir(self, base: Path, name: str) -> Path:
        skill_dir = base / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill.\n---\n\nInstructions."
        )
        return skill_dir

    def test_prefers_project_local_skill_over_cached_skill(
        self, tmp_path: Path
    ) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._make_skill_dir(skills_dir, "my-skill")

        cache_dir = tmp_path / "cache" / "acme" / "my-skill"
        cache_dir.mkdir(parents=True)
        (cache_dir / "SKILL.md").write_text(
            "---\nname: cached-skill\ndescription: Cached.\n---\n\nCached instructions."
        )
        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = cache_dir

        with (
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import resolve_registry_ref

            skill = resolve_registry_ref("@acme/my-skill")

        assert skill.name == "my-skill"

    def test_downloads_and_caches_uncached_skill_in_noninteractive_environment(
        self, tmp_path: Path
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        api = MagicMock()
        api.get_skill.return_value = _mock_skill_response("ghost-skill")

        with (
            patch.dict("os.environ", {"CI": "1", "CREWAI_NONINTERACTIVE": "1"}),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.auth.token.get_auth_token", return_value="saved-login"),
            patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
            patch("crewai_core.plus_api.PlusAPI", return_value=api),
        ):
            from crewai.skills.registry import resolve_registry_ref

            skill = resolve_registry_ref("@acme/ghost-skill")

        assert skill.name == "ghost-skill"
        api.get_skill.assert_called_once_with("acme", "ghost-skill")
        assert cache.get_cached_path("acme", "ghost-skill") == (
            tmp_path / "cache" / "acme" / "ghost-skill"
        )

    def test_resolves_cached_skill_when_project_local_skill_is_missing(
        self, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "acme" / "cached-skill"
        cache_dir.mkdir(parents=True)
        (cache_dir / "SKILL.md").write_text(
            "---\nname: cached-skill\ndescription: Cached.\n---\n\nCached instructions."
        )

        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = cache_dir

        with (
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import resolve_registry_ref

            skill = resolve_registry_ref("@acme/cached-skill")

        assert skill.name == "cached-skill"


class TestDownloadSkillAuthentication:
    def test_user_pat_takes_precedence_over_platform_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        api = MagicMock()
        api.get_skill.return_value = _mock_skill_response("my-skill")
        monkeypatch.setenv("CREWAI_USER_PAT", "user-pat")
        monkeypatch.delenv("CREWAI_ORGANIZATION_UUID", raising=False)

        with (
            platform_context("platform-token"),
            patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            download_skill("acme", "my-skill")

        plus_api.assert_called_once_with(api_key="user-pat", organization_id=None)

    def test_uses_platform_token_when_user_pat_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        api = MagicMock()
        api.get_skill.return_value = _mock_skill_response("my-skill")
        monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
        monkeypatch.delenv("CREWAI_ORGANIZATION_UUID", raising=False)

        with (
            platform_context("platform-token"),
            patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            download_skill("acme", "my-skill")

        plus_api.assert_called_once_with(api_key="platform-token", organization_id=None)

    def test_uses_user_pat_and_organization_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        api = MagicMock()
        api.get_skill.return_value = _mock_skill_response("my-skill")
        monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", raising=False)
        monkeypatch.setenv("CREWAI_USER_PAT", "user-pat")
        monkeypatch.setenv("CREWAI_ORGANIZATION_UUID", "organization-uuid")

        with (
            patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            download_skill("acme", "my-skill")

        plus_api.assert_called_once_with(
            api_key="user-pat", organization_id="organization-uuid"
        )

    def test_uses_saved_cli_login_when_runtime_tokens_are_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        api = MagicMock()
        api.get_skill.return_value = _mock_skill_response("my-skill")
        monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
        monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", raising=False)
        monkeypatch.delenv("CREWAI_ORGANIZATION_UUID", raising=False)

        with (
            patch("crewai.auth.token.get_auth_token", return_value="saved-login"),
            patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            download_skill("acme", "my-skill")

        plus_api.assert_called_once_with(api_key="saved-login", organization_id=None)

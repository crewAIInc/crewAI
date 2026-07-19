"""Tests for SkillRegistry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai.skills.registry import (
    SkillNotCachedError,
    is_registry_ref,
    parse_registry_ref,
)
import pytest


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
    """Test resolution order and CI mode behaviour."""

    def _make_skill_dir(self, base: Path, name: str) -> Path:
        """Write a minimal SKILL.md into base/name/."""
        skill_dir = base / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill.\n---\n\nInstructions."
        )
        return skill_dir

    def test_resolves_project_local(self, tmp_path: Path) -> None:
        """Local ./skills/{name}/ takes priority over cache."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        self._make_skill_dir(skills_dir, "my-skill")

        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = None

        with (
            patch("crewai.skills.registry._is_noninteractive", return_value=False),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import resolve_registry_ref
            skill = resolve_registry_ref("@acme/my-skill")

        assert skill.name == "my-skill"

    def test_raises_in_ci_when_not_cached(self, tmp_path: Path) -> None:
        """In CI mode, raise SkillNotCachedError if no local or cached copy."""
        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = None

        with (
            patch("crewai.skills.registry._is_noninteractive", return_value=True),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import resolve_registry_ref
            with pytest.raises(SkillNotCachedError) as exc_info:
                resolve_registry_ref("@acme/ghost-skill")
            assert "@acme/ghost-skill" in str(exc_info.value)

    def test_resolves_from_cache(self, tmp_path: Path) -> None:
        """Falls back to global cache when no project-local skill exists."""
        cache_dir = tmp_path / "acme" / "cached-skill"
        cache_dir.mkdir(parents=True)
        (cache_dir / "SKILL.md").write_text(
            "---\nname: cached-skill\ndescription: Cached.\n---\n\nCached instructions."
        )

        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = cache_dir

        # tmp_path has no ./skills/ directory
        with (
            patch("crewai.skills.registry._is_noninteractive", return_value=False),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import resolve_registry_ref
            skill = resolve_registry_ref("@acme/cached-skill")

        assert skill.name == "cached-skill"

    def test_skill_not_cached_error_contains_ref(self) -> None:
        err = SkillNotCachedError("@foo/bar")
        assert "@foo/bar" in str(err)
        assert err.ref == "@foo/bar"

    def test_noninteractive_downloads_when_hook_set(self, tmp_path: Path) -> None:
        """When _create_plus_client_hook is set, non-interactive mode attempts download."""
        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = None

        mock_download = MagicMock()
        mock_download.return_value = MagicMock(name="downloaded-skill")

        with (
            patch("crewai.skills.registry._is_noninteractive", return_value=True),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
            patch("crewai.skills.registry._create_plus_client_hook", new=MagicMock),
            patch("crewai.skills.registry.download_skill", mock_download),
        ):
            from crewai.skills.registry import resolve_registry_ref

            result = resolve_registry_ref("@acme/remote-skill")

        mock_download.assert_called_once_with("acme", "remote-skill", source=None)
        assert result == mock_download.return_value

    def test_noninteractive_raises_without_hook(self, tmp_path: Path) -> None:
        """Without _create_plus_client_hook, non-interactive mode still raises."""
        mock_cache = MagicMock()
        mock_cache.get_cached_path.return_value = None

        with (
            patch("crewai.skills.registry._is_noninteractive", return_value=True),
            patch.object(Path, "cwd", return_value=tmp_path),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
            patch("crewai.skills.registry._create_plus_client_hook", new=None),
        ):
            from crewai.skills.registry import resolve_registry_ref

            with pytest.raises(SkillNotCachedError):
                resolve_registry_ref("@acme/ghost-skill")


class TestDownloadSkillUsesHook:
    """Test that download_skill respects _create_plus_client_hook."""

    def test_uses_hook_client(self, tmp_path: Path) -> None:
        """download_skill uses the hook-provided client when available."""
        mock_api = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "latest_version": "1.0.0",
            "file": "," + __import__("base64").b64encode(b"").decode(),
        }
        mock_response.raise_for_status = MagicMock()
        mock_api.get_skill.return_value = mock_response

        mock_cache = MagicMock()
        skill_dir = tmp_path / "acme" / "hooked-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: hooked-skill\ndescription: Hooked.\n---\n\nHooked instructions."
        )
        mock_cache.store.return_value = skill_dir

        with (
            patch("crewai.skills.registry._create_plus_client_hook", new=lambda: mock_api),
            patch("crewai.skills.registry.SkillCacheManager", return_value=mock_cache),
        ):
            from crewai.skills.registry import download_skill

            skill = download_skill("acme", "hooked-skill")

        mock_api.get_skill.assert_called_once_with("acme", "hooked-skill")
        assert skill.name == "hooked-skill"

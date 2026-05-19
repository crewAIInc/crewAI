"""Tests for SkillRegistry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.skills.registry import (
    SkillNotCachedError,
    is_registry_ref,
    parse_registry_ref,
)


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
        with pytest.raises(ValueError, match="non-empty org and name"):
            parse_registry_ref("@/my-skill")

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty org and name"):
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

        # Mock SkillCacheManager to return None (not cached) so only local is hit
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

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from crewai.context import platform_context
from crewai.skills.cache import SkillCacheManager
from crewai.skills.models import METADATA
from crewai.skills.registry import (
    SkillRef,
    download_skill,
    is_registry_ref,
    parse_registry_ref,
    parse_skill_ref,
    resolve_registry_ref,
)
import pytest


def _skill_archive(name: str, version: str | None = None) -> bytes:
    metadata = f"metadata:\n  version: {version}\n" if version else ""
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "SKILL.md",
            f"---\nname: {name}\ndescription: Test skill.\n{metadata}---\n\nInstructions.",
        )
    return archive.getvalue()


def _skill_response(name: str, version: str | None = None) -> MagicMock:
    response = MagicMock()
    payload: dict[str, object] = {
        "latest_version": "1.0.0",
        "file": base64.b64encode(_skill_archive(name, version)).decode(),
    }
    if version is not None:
        payload["version"] = version
    response.json.return_value = payload
    return response


# Retained under its original name for the pre-existing tests below.
_mock_skill_response = _skill_response


def _stub_api(name: str, version: str | None = None) -> MagicMock:
    """A CrewAI AMP client stub serving one skill."""
    api = MagicMock()
    api.get_skill.return_value = _skill_response(name, version)
    return api


def _cache(tmp_path: Path) -> SkillCacheManager:
    return SkillCacheManager(cache_root=tmp_path / "cache")


def _write_local_skill(tmp_path: Path, name: str, version: str | None = None) -> Path:
    """Write a project-local ./skills/<name>/SKILL.md under *tmp_path*."""
    skill_dir = tmp_path / "skills" / name
    skill_dir.mkdir(parents=True)
    metadata = f"metadata:\n  version: {version}\n" if version else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill.\n{metadata}---\n\nInstructions."
    )
    return skill_dir


def _install_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    """Install *client* the way a hosted runtime does."""
    from crewai.utilities import agent_utils

    monkeypatch.setattr(agent_utils, "_create_plus_client_hook", lambda: client)


@contextmanager
def _resolving(
    tmp_path: Path, api: MagicMock, cache: SkillCacheManager | None = None
) -> Iterator[SkillCacheManager]:
    """Resolve refs against a temp cwd and cache, with *api* as the registry."""
    cache = cache or _cache(tmp_path)
    with (
        patch.object(Path, "cwd", return_value=tmp_path),
        patch("crewai.auth.token.get_auth_token", return_value="saved-login"),
        patch("crewai.skills.registry.SkillCacheManager", return_value=cache),
        patch("crewai_core.plus_api.PlusAPI", return_value=api),
    ):
        yield cache


@pytest.fixture(autouse=True)
def _no_installed_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a client installed by one test from leaking into the next."""
    from crewai.utilities import agent_utils

    monkeypatch.setattr(agent_utils, "_create_plus_client_hook", None, raising=False)


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

    def test_drops_the_version_pin(self) -> None:
        assert parse_registry_ref("@acme/my-skill@1.2.0") == ("acme", "my-skill")


class TestParseSkillRef:
    def test_without_version(self) -> None:
        assert parse_skill_ref("@acme/my-skill") == SkillRef("acme", "my-skill", None)

    def test_with_version(self) -> None:
        assert parse_skill_ref("@acme/my-skill@1.2.0") == SkillRef(
            "acme", "my-skill", "1.2.0"
        )

    def test_with_v_prefixed_version(self) -> None:
        assert parse_skill_ref("@acme/my-skill@v0.1").version == "v0.1"

    def test_with_uuid_org(self) -> None:
        assert parse_skill_ref(
            "@548e8ab5-f806-4c44-9dea-087ea05880f1/crewai-brand@2.0"
        ) == SkillRef("548e8ab5-f806-4c44-9dea-087ea05880f1", "crewai-brand", "2.0")

    def test_empty_version(self) -> None:
        with pytest.raises(ValueError, match="version must be non-empty"):
            parse_skill_ref("@acme/my-skill@")

    def test_whitespace_only_version(self) -> None:
        with pytest.raises(ValueError, match="version must be non-empty"):
            parse_skill_ref("@acme/my-skill@   ")

    def test_rejects_more_than_one_version_pin(self) -> None:
        with pytest.raises(ValueError, match="single version pin"):
            parse_skill_ref("@acme/my-skill@1.2.0@extra")

    def test_strips_surrounding_whitespace_from_the_version(self) -> None:
        assert parse_skill_ref("@acme/my-skill@ 1.2.0 ").version == "1.2.0"

    def test_round_trips_through_str(self) -> None:
        assert str(parse_skill_ref("@acme/my-skill@1.2.0")) == "@acme/my-skill@1.2.0"
        assert str(parse_skill_ref("@acme/my-skill")) == "@acme/my-skill"


class TestResolveRegistryRef:
    def test_can_resolve_metadata_without_loading_instructions(
        self, tmp_path: Path
    ) -> None:
        _write_local_skill(tmp_path, "my-skill")

        with patch.object(Path, "cwd", return_value=tmp_path):
            skill = resolve_registry_ref("@acme/my-skill", activate=False)

        assert skill.disclosure_level == METADATA
        assert skill.instructions is None

    def test_prefers_project_local_skill_over_cached_skill(
        self, tmp_path: Path
    ) -> None:
        _write_local_skill(tmp_path, "my-skill")

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


class TestResolveVersionPinnedRef:
    def test_requests_the_pinned_version_from_the_registry(self, tmp_path: Path) -> None:
        api = _stub_api("pinned-skill", "1.0.0")

        with _resolving(tmp_path, api):
            skill = resolve_registry_ref("@acme/pinned-skill@1.0.0")

        assert skill.name == "pinned-skill"
        api.get_skill.assert_called_once_with("acme", "pinned-skill", version="1.0.0")

    def test_unpinned_ref_omits_the_version_argument(self, tmp_path: Path) -> None:
        api = _stub_api("floating-skill")

        with _resolving(tmp_path, api):
            resolve_registry_ref("@acme/floating-skill")

        api.get_skill.assert_called_once_with("acme", "floating-skill")

    def test_resolves_a_pinned_skill_from_the_cache(self, tmp_path: Path) -> None:
        """Skills need not declare metadata.version; the cache records what it
        stored, so a second resolution must not download again."""
        api = _stub_api("cached-skill")

        with _resolving(tmp_path, api):
            resolve_registry_ref("@acme/cached-skill@1.0.0")
            skill = resolve_registry_ref("@acme/cached-skill@1.0.0")

        assert skill.name == "cached-skill"
        assert api.get_skill.call_count == 1

    def test_refuses_a_version_the_registry_served_instead_of_the_pin(
        self, tmp_path: Path
    ) -> None:
        """A registry predating pinning ignores the parameter and serves its
        newest version; caching that under the pin would poison every later
        lookup for it."""
        cache = _cache(tmp_path)
        api = _stub_api("drifting-skill", "2.0.0")

        with _resolving(tmp_path, api, cache=cache), pytest.raises(
            RuntimeError, match="served version '2.0.0' rather than the pinned"
        ):
            resolve_registry_ref("@acme/drifting-skill@1.0.0")

        assert cache.get_cached_path("acme", "drifting-skill") is None

    def test_redownloads_when_the_cached_version_is_not_the_pinned_one(
        self, tmp_path: Path
    ) -> None:
        cache = SkillCacheManager(cache_root=tmp_path / "cache")
        cache.store("acme", "drifting-skill", "1.0.0", _skill_archive("drifting-skill"))
        api = _stub_api("drifting-skill", "2.0.0")

        with _resolving(tmp_path, api, cache=cache):
            resolve_registry_ref("@acme/drifting-skill@2.0.0")

        api.get_skill.assert_called_once_with("acme", "drifting-skill", version="2.0.0")

    def test_uses_a_project_local_skill_that_declares_the_pinned_version(
        self, tmp_path: Path
    ) -> None:
        _write_local_skill(tmp_path, "local-skill", "1.0.0")
        api = _stub_api("local-skill")

        with _resolving(tmp_path, api):
            skill = resolve_registry_ref("@acme/local-skill@v1.0.0")

        assert skill.name == "local-skill"
        api.get_skill.assert_not_called()

    def test_downloads_when_a_project_local_skill_declares_another_version(
        self, tmp_path: Path
    ) -> None:
        _write_local_skill(tmp_path, "stale-skill", "1.0.0")
        api = _stub_api("stale-skill", "2.0.0")

        with _resolving(tmp_path, api):
            resolve_registry_ref("@acme/stale-skill@2.0.0")

        api.get_skill.assert_called_once_with("acme", "stale-skill", version="2.0.0")


class TestDownloadSkillClient:
    """A hosted runtime has no user credential, so its own client must be used."""

    def test_uses_the_client_the_runtime_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed = _stub_api("my-skill")
        monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
        _install_client(monkeypatch, installed)

        with (
            patch("crewai.skills.registry.SkillCacheManager", return_value=_cache(tmp_path)),
            patch("crewai.auth.token.get_auth_token") as get_auth_token,
            patch("crewai_core.plus_api.PlusAPI") as plus_api,
        ):
            download_skill("acme", "my-skill")

        installed.get_skill.assert_called_once_with("acme", "my-skill")
        # No user credential is read at all when a client is installed.
        plus_api.assert_not_called()
        get_auth_token.assert_not_called()

    def test_awaits_an_async_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = _skill_response("my-skill")

        async def get_skill(org: str, name: str, **kwargs: object) -> MagicMock:
            return response

        installed = MagicMock()
        installed.get_skill = get_skill
        _install_client(monkeypatch, installed)

        with patch(
            "crewai.skills.registry.SkillCacheManager", return_value=_cache(tmp_path)
        ):
            skill = download_skill("acme", "my-skill")

        assert skill.name == "my-skill"

    @pytest.mark.parametrize(
        "get_skill", [None, "not-callable"], ids=["missing", "not-callable"]
    )
    def test_falls_back_when_the_client_cannot_fetch_skills(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, get_skill: object
    ) -> None:
        api = _stub_api("my-skill")
        monkeypatch.setenv("CREWAI_USER_PAT", "user-pat")
        installed = MagicMock(spec=[] if get_skill is None else ["get_skill"])
        if get_skill is not None:
            installed.get_skill = get_skill
        _install_client(monkeypatch, installed)

        with (
            patch("crewai.skills.registry.SkillCacheManager", return_value=_cache(tmp_path)),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            skill = download_skill("acme", "my-skill")

        assert skill.name == "my-skill"
        plus_api.assert_called_once_with(api_key="user-pat", organization_id=None)

    def test_falls_back_when_the_client_cannot_forward_a_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repository agents pin automatically, so a client whose get_skill
        predates pinning must not turn a working lookup into a TypeError."""
        api = _stub_api("my-skill", "1.0.0")
        monkeypatch.setenv("CREWAI_USER_PAT", "user-pat")

        class VersionUnawareClient:
            def get_skill(self, org: str, name: str) -> MagicMock:
                raise AssertionError("must not be called with a pin")

        _install_client(monkeypatch, VersionUnawareClient())

        with (
            patch("crewai.skills.registry.SkillCacheManager", return_value=_cache(tmp_path)),
            patch("crewai_core.plus_api.PlusAPI", return_value=api) as plus_api,
        ):
            skill = download_skill("acme", "my-skill", version="1.0.0")

        assert skill.name == "my-skill"
        plus_api.assert_called_once_with(api_key="user-pat", organization_id=None)

    def test_uses_a_version_unaware_client_for_unpinned_refs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str]] = []

        class VersionUnawareClient:
            def get_skill(self, org: str, name: str) -> MagicMock:
                calls.append((org, name))
                return _skill_response("my-skill")

        _install_client(monkeypatch, VersionUnawareClient())

        with patch(
            "crewai.skills.registry.SkillCacheManager", return_value=_cache(tmp_path)
        ):
            download_skill("acme", "my-skill")

        assert calls == [("acme", "my-skill")]

    @pytest.mark.parametrize("version", ["", "   "], ids=["empty", "whitespace"])
    def test_rejects_a_blank_pin_rather_than_floating_to_latest(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        installed = _stub_api("my-skill")
        _install_client(monkeypatch, installed)

        with pytest.raises(ValueError, match="must be non-empty"):
            download_skill("acme", "my-skill", version=version)

        installed.get_skill.assert_not_called()

    def test_reports_the_pinned_ref_when_the_download_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed = MagicMock()
        installed.get_skill.side_effect = RuntimeError("401 Unauthorized")
        _install_client(monkeypatch, installed)

        with pytest.raises(
            RuntimeError, match=r"Failed to download skill '@acme/my-skill@1\.0\.0'"
        ):
            download_skill("acme", "my-skill", version="1.0.0")


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

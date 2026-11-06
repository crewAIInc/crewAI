"""A failed registry download must not destroy an installed global skill."""

from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch
import zipfile

import crewai.skills.cache as cache_module
from crewai.skills.cache import SkillCacheManager
from crewai_cli.skills.main import SkillCommand
import httpx
import pytest


Installer = Callable[[str, bytes], Path]


def _archive(files: dict[str, str], archive_format: str = "tar") -> bytes:
    """Build real archives, including members that extraction may reject."""
    buffer = io.BytesIO()
    if archive_format == "zip":
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
    else:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, text in files.items():
                data = text.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture(params=["sdk", "cli-with-sdk", "cli-without-sdk"])
def cache_install(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Installer, Path]:
    """Exercise the public entry points with an offline registry response."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cache_root = tmp_path / ".crewai" / "skills"
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", cache_root)
    destination = cache_root / "acme" / "my-skill"
    if request.param == "sdk":
        cache = SkillCacheManager(cache_root=cache_root)
        return lambda version, data: cache.store(
            "acme", "my-skill", version, data
        ), destination

    monkeypatch.setattr("crewai_cli.command.get_auth_token", lambda: "offline-token")
    monkeypatch.setattr(SkillCommand, "_print_current_organization", lambda self: None)
    command = SkillCommand()

    def install(version: str, data: bytes) -> Path:
        response = httpx.Response(
            200,
            json={"version": version, "file": base64.b64encode(data).decode()},
        )
        monkeypatch.setattr(
            command.plus_api_client, "get_skill", lambda *args: response
        )
        context = (
            patch.dict(sys.modules, {"crewai.skills.cache": None})
            if request.param == "cli-without-sdk"
            else nullcontext()
        )
        with context:
            command.install("@acme/my-skill")
        return destination

    return install, destination


@pytest.mark.parametrize("invalid_archive", ["unreadable", "rejected-tar"])
def test_failed_update_preserves_installed_files_and_metadata(
    cache_install: tuple[Installer, Path], invalid_archive: str
) -> None:
    install, destination = cache_install
    install("1.0.0", _archive({"SKILL.md": "working instructions", "old.txt": "keep"}))
    original = {path.name: path.read_bytes() for path in destination.iterdir()}
    broken = (
        b"not an archive"
        if invalid_archive == "unreadable"
        else _archive({"SKILL.md": "partial update", "../outside.txt": "reject"})
    )

    with pytest.raises((ValueError, tarfile.TarError, zipfile.BadZipFile)):
        install("2.0.0", broken)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original
    assert list(destination.parent.iterdir()) == [destination]
    assert not (destination.parent / "outside.txt").exists()


def test_failed_first_install_leaves_no_partial_cache_entry(
    cache_install: tuple[Installer, Path],
) -> None:
    install, destination = cache_install

    with pytest.raises(zipfile.BadZipFile):
        install("1.0.0", b"not an archive")

    assert not destination.exists()
    assert not list(destination.parent.iterdir())


@pytest.mark.parametrize("archive_format", ["tar", "zip"])
def test_successful_update_replaces_files_and_records_the_version(
    cache_install: tuple[Installer, Path], archive_format: str
) -> None:
    install, destination = cache_install
    install("1.0.0", _archive({"SKILL.md": "old", "old.txt": "stale"}))

    result = install("2.0.0", _archive({"SKILL.md": "new"}, archive_format))

    assert result == destination
    assert (destination / "SKILL.md").read_text() == "new"
    assert not (destination / "old.txt").exists()
    assert (
        json.loads((destination / ".crewai_meta.json").read_text())["version"]
        == "2.0.0"
    )
    assert list(destination.parent.iterdir()) == [destination]

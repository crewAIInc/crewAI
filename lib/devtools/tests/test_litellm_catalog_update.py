"""The release catalog refresh downloads LiteLLM's snapshot with curl."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from crewai_devtools.litellm_catalog_update import CATALOG_URL, update_catalog
import pytest


def _destination(repo_root: Path) -> Path:
    return (
        repo_root
        / "lib/crewai/src/crewai/llms/litellm_catalog/model_prices_and_context_window.json"
    )


def test_update_catalog_replaces_the_snapshot_after_a_valid_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text('{"old": true}\n', encoding="utf-8")
    seen: list[list[str]] = []

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append(args)
        output = Path(args[args.index("--output") + 1])
        output.write_bytes(b'{"sample_spec": {"max_input_tokens": 1}}')
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "crewai_devtools.litellm_catalog_update.shutil.which",
        lambda _name: "/usr/bin/curl",
    )
    monkeypatch.setattr(
        "crewai_devtools.litellm_catalog_update.subprocess.run", fake_run
    )

    written = update_catalog(tmp_path)

    assert written == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "sample_spec": {"max_input_tokens": 1}
    }
    assert seen[0][0] == "/usr/bin/curl"
    assert CATALOG_URL in seen[0]
    assert not list(destination.parent.glob(".model_prices_and_context_window.*.json"))


def test_update_catalog_keeps_the_existing_snapshot_when_the_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = _destination(tmp_path)
    destination.parent.mkdir(parents=True)
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fake_run(
        args: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 22, stdout="", stderr="curl: (22) 404")

    monkeypatch.setattr(
        "crewai_devtools.litellm_catalog_update.shutil.which",
        lambda _name: "/usr/bin/curl",
    )
    monkeypatch.setattr(
        "crewai_devtools.litellm_catalog_update.subprocess.run", fake_run
    )

    with pytest.raises(RuntimeError, match="404"):
        update_catalog(tmp_path)

    assert destination.read_text(encoding="utf-8") == '{"old": true}\n'
    assert not list(destination.parent.glob(".model_prices_and_context_window.*.json"))

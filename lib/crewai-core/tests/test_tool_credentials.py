import os
from pathlib import Path
from types import SimpleNamespace

from crewai_core import tool_credentials
import pytest


@pytest.mark.parametrize(
    ("source", "handles"),
    [
        ('{ index = "private-tools" }', ["PRIVATE_TOOLS"]),
        ('[{ index = "private-tools" }]', ["PRIVATE_TOOLS"]),
        (
            '[{ index = "private-linux", marker = "sys_platform == \'linux\'" },'
            '{ index = "private-mac", marker = "sys_platform == \'darwin\'" }]',
            ["PRIVATE_LINUX", "PRIVATE_MAC"],
        ),
        (
            '[{ index = "private-tools", marker = "sys_platform == \'linux\'" },'
            '{ path = "../local-tool", marker = "sys_platform != \'linux\'" }]',
            ["PRIVATE_TOOLS"],
        ),
        ('{ path = "../local-tool" }', []),
    ],
    ids=["single", "singleton-list", "platform-list", "mixed-list", "local-path"],
)
def test_credentials_for_uv_source_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    handles: list[str],
) -> None:
    """Collect credentials for every referenced index in real TOML sources."""
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.uv.sources]\ncustom-tool = {source}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tool_credentials,
        "Settings",
        lambda: SimpleNamespace(
            tool_repository_username="test-user",
            tool_repository_password="test-password",
        ),
    )
    for handle in ["PRIVATE_TOOLS", "PRIVATE_LINUX", "PRIVATE_MAC"]:
        monkeypatch.delenv(f"UV_INDEX_{handle}_USERNAME", raising=False)
        monkeypatch.delenv(f"UV_INDEX_{handle}_PASSWORD", raising=False)
    monkeypatch.setenv("CREWAI_TEST_PARENT_ENV", "preserved")

    env = tool_credentials.build_env_with_all_tool_credentials()

    expected = {}
    for handle in handles:
        expected[f"UV_INDEX_{handle}_USERNAME"] = "test-user"
        expected[f"UV_INDEX_{handle}_PASSWORD"] = "test-password"
    credential_keys = [
        f"UV_INDEX_{handle}_{suffix}"
        for handle in ["PRIVATE_TOOLS", "PRIVATE_LINUX", "PRIVATE_MAC"]
        for suffix in ["USERNAME", "PASSWORD"]
    ]
    assert {key: env[key] for key in credential_keys if key in env} == expected
    assert env["CREWAI_TEST_PARENT_ENV"] == "preserved"
    assert all(key not in os.environ for key in credential_keys)

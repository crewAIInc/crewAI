"""Guardrails for security-sensitive dependency pins in crewai's pyproject.

These tests parse the published requirement specifiers so that a stale pin
cannot silently reintroduce a known-vulnerable dependency range.
"""

from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT_PATH = Path(__file__).parents[2] / "pyproject.toml"


def _get_requirement(name: str) -> Requirement:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    for spec in data["project"]["dependencies"]:
        requirement = Requirement(spec)
        if requirement.name == name:
            return requirement
    raise AssertionError(f"{name} not found in [project.dependencies]")


class TestMcpPin:
    """GHSA-hvrp-rf83-w775 / GHSA-jpw9-pfvf-9f58 (fixed in mcp 1.27.2) and
    GHSA-vj7q-gjh5-988w (fixed in 1.28.1): session/task-handler auth gaps and
    missing WebSocket Host/Origin validation in the MCP Python SDK. A
    compatible-release pin on 1.26.x blocks all three patches."""

    def test_requirement_excludes_vulnerable_range(self) -> None:
        requirement = _get_requirement("mcp")
        for vulnerable in ("1.26.0", "1.27.2", "1.28.0"):
            assert not requirement.specifier.contains(vulnerable), (
                f"mcp requirement '{requirement}' admits {vulnerable}, which "
                "is vulnerable to GHSA-vj7q-gjh5-988w (fixed in 1.28.1)"
            )

    def test_requirement_admits_patched_releases(self) -> None:
        requirement = _get_requirement("mcp")
        assert requirement.specifier.contains("1.28.1"), (
            f"mcp requirement '{requirement}' blocks patched version 1.28.1"
        )


class TestJsonRepairPin:
    """GHSA-xf7x-x43h-rpqh: json-repair < 0.60.1 has an unbounded-CPU DoS on
    circular JSON Schema $ref structures. crewai feeds LLM output to
    json_repair, so the vulnerable range is a real DoS surface (OSS-91)."""

    def test_requirement_excludes_vulnerable_range(self) -> None:
        requirement = _get_requirement("json-repair")
        for vulnerable in ("0.25.2", "0.25.3", "0.59.0", "0.60.0"):
            assert not requirement.specifier.contains(vulnerable), (
                f"json-repair requirement '{requirement}' admits {vulnerable}, "
                "which is vulnerable to GHSA-xf7x-x43h-rpqh (fixed in 0.60.1)"
            )

    def test_requirement_admits_patched_releases(self) -> None:
        requirement = _get_requirement("json-repair")
        for patched in ("0.60.1", "0.61.4"):
            assert requirement.specifier.contains(patched), (
                f"json-repair requirement '{requirement}' blocks patched "
                f"version {patched}"
            )

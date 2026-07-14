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
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    for spec in data["project"]["dependencies"]:
        requirement = Requirement(spec)
        if requirement.name == name:
            return requirement
    raise AssertionError(f"{name} not found in [project.dependencies]")


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

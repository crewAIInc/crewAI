"""Guardrail for the chromadb requirement specifier (OSS-90).

CVE-2026-45829 is a pre-auth code injection in the ChromaDB server with no
patched release yet (vulnerable range >=1.0.0, <=1.5.9). A compatible-release
pin like ~=1.1.0 would silently block the 1.6+ patch the moment it ships, so
the requirement must keep the 1.x line open.
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


class TestChromadbPin:
    def test_requirement_does_not_block_future_1x_patch(self) -> None:
        requirement = _get_requirement("chromadb")
        for candidate in ("1.6.0", "1.9.0"):
            assert requirement.specifier.contains(candidate), (
                f"chromadb requirement '{requirement}' blocks {candidate}; a "
                "CVE-2026-45829 patch release (>1.5.9) must be installable "
                "without waiting for a crewai release"
            )

    def test_requirement_keeps_known_compatible_floor(self) -> None:
        requirement = _get_requirement("chromadb")
        assert requirement.specifier.contains("1.1.0"), (
            f"chromadb requirement '{requirement}' dropped the 1.1.0 floor"
        )
        assert not requirement.specifier.contains("2.0.0"), (
            f"chromadb requirement '{requirement}' admits an untested major"
        )

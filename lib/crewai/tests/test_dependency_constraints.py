"""Tests for dependency version constraints.

Regression tests to ensure critical dependency constraints are correct,
particularly for packages whose older versions have broken metadata.
"""

import importlib.metadata
from pathlib import Path

import tomli


def _read_crewai_pyproject() -> dict:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)


class TestTokenizersDependency:
    """Regression tests for tokenizers dependency (issue #4550).

    tokenizers 0.20.x has a broken pyproject.toml (missing project.version),
    which causes installation failures when building from source (sdist).
    The constraint must require >= 0.21 to avoid the broken versions.
    """

    def test_tokenizers_constraint_excludes_broken_versions(self):
        pyproject = _read_crewai_pyproject()
        deps = pyproject["project"]["dependencies"]
        tokenizers_dep = next(
            (d for d in deps if d.startswith("tokenizers")), None
        )
        assert tokenizers_dep is not None, "tokenizers dependency not found in pyproject.toml"
        assert "0.20" not in tokenizers_dep, (
            f"tokenizers constraint '{tokenizers_dep}' still allows 0.20.x which has a broken sdist"
        )

    def test_tokenizers_constraint_allows_recent_versions(self):
        pyproject = _read_crewai_pyproject()
        deps = pyproject["project"]["dependencies"]
        tokenizers_dep = next(
            (d for d in deps if d.startswith("tokenizers")), None
        )
        assert tokenizers_dep is not None
        assert ">=0.21" in tokenizers_dep, (
            f"tokenizers constraint '{tokenizers_dep}' should require >=0.21"
        )

    def test_tokenizers_is_importable(self):
        import tokenizers

        assert tokenizers is not None

    def test_installed_tokenizers_version_is_not_broken(self):
        version = importlib.metadata.version("tokenizers")
        major, minor = (int(x) for x in version.split(".")[:2])
        assert (major, minor) >= (0, 21), (
            f"Installed tokenizers {version} is from the 0.20.x range with broken sdist metadata"
        )

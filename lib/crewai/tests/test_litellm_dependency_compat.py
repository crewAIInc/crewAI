"""Tests for litellm optional dependency compatibility with crewAI core deps.

Regression tests for https://github.com/crewAIInc/crewAI/issues/6089:
the litellm extra must not pin versions whose transitive dependencies
conflict with crewAI's own requirements (openai, python-dotenv, etc.).
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    """Load lib/crewai/pyproject.toml as a dict."""
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def _parse_specifier(dep_string: str) -> tuple[str, SpecifierSet]:
    """Split 'pkg>=1.2,<3' into ('pkg', SpecifierSet('>=1.2,<3'))."""
    for op in (">=", "<=", "~=", "==", "!=", ">", "<"):
        idx = dep_string.find(op)
        if idx != -1:
            name = dep_string[:idx].strip()
            spec = dep_string[idx:].split(";")[0].strip()
            return name, SpecifierSet(spec)
    return dep_string.strip(), SpecifierSet()


class TestLitellmDependencyBounds:
    """Verify the litellm optional-dependency range is resolvable."""

    def test_litellm_extra_exists(self, pyproject):
        """The litellm optional dependency group must be defined."""
        opt_deps = pyproject["project"]["optional-dependencies"]
        assert "litellm" in opt_deps, "Missing 'litellm' optional-dependency group"

    def test_litellm_lower_bound_at_least_1_84(self, pyproject):
        """litellm lower bound must be >= 1.84.0 to avoid transitive conflicts.

        Versions < 1.84 pin openai and python-dotenv to ranges that are
        incompatible with crewAI's core requirements.
        """
        opt_deps = pyproject["project"]["optional-dependencies"]
        litellm_deps = opt_deps["litellm"]
        litellm_dep = [d for d in litellm_deps if d.startswith("litellm")]
        assert litellm_dep, "No litellm dependency found in litellm extras"

        _, spec = _parse_specifier(litellm_dep[0])

        # 1.83.x versions have conflicting transitive deps; must be excluded
        assert not spec.contains(Version("1.83.7")), (
            "litellm 1.83.7 must be excluded (pins python-dotenv==1.0.1)"
        )
        assert not spec.contains(Version("1.83.8")), (
            "litellm 1.83.8 must be excluded (pins openai==2.24.0)"
        )
        assert not spec.contains(Version("1.83.14")), (
            "litellm 1.83.14 must be excluded (pins openai==2.24.0)"
        )

        # 1.84.0+ relaxes transitive pins to compatible bounds
        assert spec.contains(Version("1.84.0")), (
            "litellm 1.84.0 should be allowed (compatible transitive deps)"
        )

    def test_litellm_upper_bound_allows_recent_versions(self, pyproject):
        """litellm range must accept recent 1.x releases."""
        opt_deps = pyproject["project"]["optional-dependencies"]
        litellm_deps = opt_deps["litellm"]
        litellm_dep = [d for d in litellm_deps if d.startswith("litellm")]

        _, spec = _parse_specifier(litellm_dep[0])

        # Ensure reasonable recent versions are included
        assert spec.contains(Version("1.87.0")), (
            "litellm 1.87.x should be allowed"
        )
        assert spec.contains(Version("1.90.0")), (
            "litellm 1.90.x should be allowed"
        )

    def test_litellm_range_excludes_v2(self, pyproject):
        """litellm range must not include v2 (potential breaking changes)."""
        opt_deps = pyproject["project"]["optional-dependencies"]
        litellm_deps = opt_deps["litellm"]
        litellm_dep = [d for d in litellm_deps if d.startswith("litellm")]

        _, spec = _parse_specifier(litellm_dep[0])

        assert not spec.contains(Version("2.0.0")), (
            "litellm 2.x should be excluded to avoid breaking changes"
        )

    def test_core_openai_dep_compatible_with_litellm_range(self, pyproject):
        """crewAI's openai requirement must be satisfiable alongside litellm>=1.84.

        litellm>=1.84.0 requires openai>=2.20.0,<3.0.0, which overlaps
        with crewAI's openai>=2.30.0,<3.
        """
        deps = pyproject["project"]["dependencies"]
        openai_deps = [d for d in deps if d.startswith("openai")]
        assert openai_deps, "openai must be a core dependency"

        _, crewai_openai_spec = _parse_specifier(openai_deps[0])

        # litellm>=1.84 allows openai>=2.20.0,<3.0.0
        # crewAI requires openai>=2.30.0,<3
        # The intersection should be non-empty
        test_version = Version("2.30.0")
        litellm_openai_spec = SpecifierSet(">=2.20.0,<3.0.0")
        assert crewai_openai_spec.contains(test_version) and litellm_openai_spec.contains(test_version), (
            "openai 2.30.0 must satisfy both crewAI and litellm>=1.84 requirements"
        )

    def test_core_python_dotenv_dep_compatible_with_litellm_range(self, pyproject):
        """crewAI's python-dotenv requirement must be satisfiable alongside litellm>=1.84.

        litellm>=1.84.0 requires python-dotenv>=1.0.0,<2.0, which overlaps
        with crewAI's python-dotenv>=1.2.2,<2.
        """
        deps = pyproject["project"]["dependencies"]
        dotenv_deps = [d for d in deps if d.startswith("python-dotenv")]
        assert dotenv_deps, "python-dotenv must be a core dependency"

        _, crewai_dotenv_spec = _parse_specifier(dotenv_deps[0])

        # litellm>=1.84 allows python-dotenv>=1.0.0,<2.0
        # crewAI requires python-dotenv>=1.2.2,<2
        # The intersection should be non-empty
        test_version = Version("1.2.2")
        litellm_dotenv_spec = SpecifierSet(">=1.0.0,<2.0")
        assert crewai_dotenv_spec.contains(test_version) and litellm_dotenv_spec.contains(test_version), (
            "python-dotenv 1.2.2 must satisfy both crewAI and litellm>=1.84 requirements"
        )

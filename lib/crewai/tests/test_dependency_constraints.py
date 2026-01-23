"""Tests for dependency constraints in pyproject.toml.

These tests verify that dependency constraints are flexible enough to allow
installation alongside other packages that may require different versions
of shared dependencies.
"""

from pathlib import Path

import pytest
import tomli


@pytest.fixture
def pyproject_dependencies():
    """Load dependencies from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["dependencies"]


def test_tokenizers_constraint_allows_newer_versions(pyproject_dependencies):
    """Test that tokenizers constraint allows versions >= 0.21.

    This test ensures that the tokenizers dependency constraint is flexible
    enough to allow installation of transformers 4.51+ which requires
    tokenizers >= 0.21.

    See: https://github.com/crewAIInc/crewAI/issues/4268
    """
    tokenizers_dep = None
    for dep in pyproject_dependencies:
        if dep.startswith("tokenizers"):
            tokenizers_dep = dep
            break

    assert tokenizers_dep is not None, "tokenizers dependency not found in pyproject.toml"

    # The constraint should use >= (minimum version) not ~= (compatible release)
    # ~=0.20.3 means >=0.20.3,<0.21.0 which blocks transformers 4.51+
    # >=0.20.3 allows any version >= 0.20.3 including 0.21+
    assert "~=" not in tokenizers_dep, (
        f"tokenizers constraint '{tokenizers_dep}' uses ~= operator which is too restrictive. "
        "This blocks transformers 4.51+ which requires tokenizers >= 0.21. "
        "Use >= instead of ~= to allow newer versions."
    )
    assert ">=" in tokenizers_dep, (
        f"tokenizers constraint '{tokenizers_dep}' should use >= operator "
        "to allow newer versions required by transformers 4.51+"
    )

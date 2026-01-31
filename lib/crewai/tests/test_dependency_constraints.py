"""Test that dependency constraints are compatible with common integrations."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_pyproject_dependencies() -> dict[str, str]:
    """Parse pyproject.toml and return dependencies as a dict."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    dependencies = {}
    for dep in data.get("project", {}).get("dependencies", []):
        if ">=" in dep or "~=" in dep or "==" in dep or "<" in dep:
            parts = dep.replace(">=", " ").replace("~=", " ").replace("==", " ").replace(",", " ").replace("<", " <").split()
            if parts:
                name = parts[0].strip()
                constraint = dep[len(name):].strip()
                dependencies[name] = constraint
    return dependencies


def test_openai_constraint_allows_openlit_compatible_versions():
    """Test that openai constraint allows versions >= 1.92.0 required by OpenLit.
    
    This test ensures that the openai dependency constraint is relaxed enough
    to allow integration with OpenLit, which requires openai >= 1.92.0.
    
    See: https://github.com/crewAIInc/crewAI/issues/4270
    """
    dependencies = get_pyproject_dependencies()
    
    assert "openai" in dependencies, "openai should be in dependencies"
    
    openai_constraint = dependencies["openai"]
    specifier = SpecifierSet(openai_constraint)
    
    openlit_required_version = Version("1.92.0")
    assert openlit_required_version in specifier, (
        f"openai constraint '{openai_constraint}' should allow version 1.92.0 "
        f"required by OpenLit. Current constraint blocks OpenLit integration."
    )
    
    higher_1x_version = Version("1.109.0")
    assert higher_1x_version in specifier, (
        f"openai constraint '{openai_constraint}' should allow higher 1.x versions"
    )


def test_openai_constraint_has_upper_bound():
    """Test that openai constraint has an upper bound to prevent breaking changes.
    
    The constraint should prevent openai 2.x which may have breaking changes.
    """
    dependencies = get_pyproject_dependencies()
    
    assert "openai" in dependencies, "openai should be in dependencies"
    
    openai_constraint = dependencies["openai"]
    specifier = SpecifierSet(openai_constraint)
    
    version_2 = Version("2.0.0")
    assert version_2 not in specifier, (
        f"openai constraint '{openai_constraint}' should NOT allow version 2.0.0 "
        f"to prevent potential breaking changes from major version bump"
    )


def test_openai_constraint_minimum_version():
    """Test that openai constraint has a reasonable minimum version."""
    dependencies = get_pyproject_dependencies()
    
    assert "openai" in dependencies, "openai should be in dependencies"
    
    openai_constraint = dependencies["openai"]
    specifier = SpecifierSet(openai_constraint)
    
    min_version = Version("1.83.0")
    assert min_version in specifier, (
        f"openai constraint '{openai_constraint}' should allow version 1.83.0"
    )
    
    old_version = Version("1.82.0")
    assert old_version not in specifier, (
        f"openai constraint '{openai_constraint}' should NOT allow version 1.82.0 "
        f"which is below the minimum required version"
    )

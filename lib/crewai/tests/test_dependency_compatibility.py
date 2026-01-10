"""Test that crewai dependencies are compatible with common integrations.

This test module verifies that crewai's dependency constraints are flexible enough
to allow installation alongside common third-party packages like opik for monitoring.

Related issue: https://github.com/crewAIInc/crewAI/issues/4201
"""

import re
from pathlib import Path

import tomli


def get_pyproject_path() -> Path:
    """Get the path to the crewai pyproject.toml file."""
    return Path(__file__).parent.parent / "pyproject.toml"


def parse_pyproject() -> dict:
    """Parse the pyproject.toml file."""
    pyproject_path = get_pyproject_path()
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)


def test_openai_dependency_is_flexible():
    """Test that openai dependency uses >= instead of ~= to allow version flexibility.

    The ~= operator (compatible release) is too restrictive and can cause dependency
    conflicts with packages like opik that also depend on openai.

    For example, openai~=1.83.0 means >=1.83.0,<1.84.0 which is very restrictive.
    Using openai>=1.13.3 allows any version >= 1.13.3 which is more flexible.
    """
    pyproject = parse_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    openai_dep = None
    for dep in dependencies:
        if dep.startswith("openai"):
            openai_dep = dep
            break

    assert openai_dep is not None, "openai dependency not found in pyproject.toml"

    # Check that it uses >= instead of ~=
    assert "~=" not in openai_dep, (
        f"openai dependency should use >= instead of ~= for flexibility. "
        f"Found: {openai_dep}"
    )
    assert ">=" in openai_dep, (
        f"openai dependency should use >= for minimum version. Found: {openai_dep}"
    )


def test_pydantic_dependency_allows_minor_updates():
    """Test that pydantic dependency allows minor version updates within v2.

    Using pydantic>=2.x.x,<3.0.0 allows minor updates while staying within v2.
    """
    pyproject = parse_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    pydantic_dep = None
    for dep in dependencies:
        if dep.startswith("pydantic") and not dep.startswith("pydantic-settings"):
            pydantic_dep = dep
            break

    assert pydantic_dep is not None, "pydantic dependency not found in pyproject.toml"

    # Check that it uses >= and <3.0.0 instead of ~=
    assert "~=" not in pydantic_dep, (
        f"pydantic dependency should use >= instead of ~= for flexibility. "
        f"Found: {pydantic_dep}"
    )
    assert ">=" in pydantic_dep, (
        f"pydantic dependency should use >= for minimum version. Found: {pydantic_dep}"
    )
    assert "<3.0.0" in pydantic_dep, (
        f"pydantic dependency should have <3.0.0 upper bound. Found: {pydantic_dep}"
    )


def test_pydantic_settings_dependency_allows_minor_updates():
    """Test that pydantic-settings dependency allows minor version updates within v2."""
    pyproject = parse_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    pydantic_settings_dep = None
    for dep in dependencies:
        if dep.startswith("pydantic-settings"):
            pydantic_settings_dep = dep
            break

    assert (
        pydantic_settings_dep is not None
    ), "pydantic-settings dependency not found in pyproject.toml"

    # Check that it uses >= and <3.0.0 instead of ~=
    assert "~=" not in pydantic_settings_dep, (
        f"pydantic-settings dependency should use >= instead of ~= for flexibility. "
        f"Found: {pydantic_settings_dep}"
    )
    assert ">=" in pydantic_settings_dep, (
        f"pydantic-settings dependency should use >= for minimum version. "
        f"Found: {pydantic_settings_dep}"
    )


def test_core_dependencies_use_flexible_constraints():
    """Test that core dependencies use >= instead of ~= for flexibility.

    The ~= operator is too restrictive for most dependencies and can cause
    conflicts with third-party packages.
    """
    pyproject = parse_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    # These are core dependencies that should use flexible constraints
    core_deps = [
        "openai",
        "pydantic",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "click",
    ]

    for core_dep in core_deps:
        matching_dep = None
        for dep in dependencies:
            if dep.startswith(core_dep):
                matching_dep = dep
                break

        if matching_dep:
            assert "~=" not in matching_dep, (
                f"{core_dep} dependency should use >= instead of ~= for flexibility. "
                f"Found: {matching_dep}"
            )


def test_no_overly_restrictive_pinning():
    """Test that dependencies don't use overly restrictive pinning.

    Dependencies should not use == (exact version) or ~= (compatible release)
    unless there's a specific reason documented.
    """
    pyproject = parse_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    for dep in dependencies:
        # Skip comments
        if dep.strip().startswith("#"):
            continue

        # Check for exact version pinning (==)
        # Allow == only if there's a known reason
        if "==" in dep:
            # Currently no dependencies should use ==
            assert False, (
                f"Dependency uses exact version pinning (==) which is too restrictive: {dep}"
            )

        # Check for compatible release (~=)
        if "~=" in dep:
            assert False, (
                f"Dependency uses compatible release (~=) which can be too restrictive: {dep}. "
                f"Consider using >= instead."
            )

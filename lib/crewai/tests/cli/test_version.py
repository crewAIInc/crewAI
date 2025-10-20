"""Test for version management."""

from crewai import __version__
from crewai.cli.version import get_crewai_version


def test_dynamic_versioning_consistency():
    """Test that dynamic versioning provides consistent version across all access methods."""
    cli_version = get_crewai_version()
    package_version = __version__

    # Both should return the same version string
    assert cli_version == package_version

    # Version should not be empty
    assert package_version is not None
    assert len(package_version.strip()) > 0

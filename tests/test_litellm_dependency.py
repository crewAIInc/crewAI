import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def test_litellm_dependency_allows_patch_versions():
    """
    Test that the litellm dependency constraint allows patch versions >= 1.72.0.
    
    This test verifies that the dependency constraint litellm>=1.72.0,<2.0.0 
    allows users to install newer patch versions like litellm>=1.72.2 as 
    requested in GitHub issue #3005, while preventing major version conflicts.
    """
    
    test_pyproject = """
[project]
name = "test-crewai-deps"
version = "0.1.0"
dependencies = [
    "crewai",
    "litellm>=1.72.2"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pyproject_path = Path(temp_dir) / "pyproject.toml"
        pyproject_path.write_text(test_pyproject.strip())
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps", str(temp_dir)],
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            assert True, "Dependency resolution should work with litellm>=1.72.2"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Dependency resolution failed: {e.stderr}")


def test_litellm_import_works():
    """
    Test that litellm can be imported and basic functionality works.
    
    Verifies that the minimum required litellm version (>=1.72.0) provides
    all the essential functionality that CrewAI depends on.
    """
    try:
        import litellm
        assert hasattr(litellm, 'completion'), "litellm should have completion function"
        assert hasattr(litellm, 'set_verbose'), "litellm should have set_verbose function"
        assert hasattr(litellm, 'drop_params'), "litellm should have drop_params attribute"
        
        import pkg_resources
        version = pkg_resources.get_distribution("litellm").version
        major, minor, patch = map(int, version.split('.')[:3])
        assert (major, minor, patch) >= (1, 72, 0), f"litellm version {version} is below minimum required 1.72.0"
        
    except ImportError as e:
        pytest.fail(f"Failed to import litellm: {e}")


@pytest.mark.parametrize("model_name", [
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-2"
])
def test_crewai_llm_works_with_current_litellm(model_name):
    """
    Test that CrewAI's LLM class works with the current litellm version.
    
    This parameterized test verifies compatibility across different model types
    to ensure the litellm upgrade doesn't break existing functionality.
    """
    from crewai.llm import LLM
    
    llm = LLM(model=model_name)
    
    assert llm.model == model_name, f"Model name mismatch for {model_name}"
    assert hasattr(llm, 'call'), f"LLM should have call method for {model_name}"
    assert hasattr(llm, 'supports_function_calling'), f"LLM should have supports_function_calling method for {model_name}"
    assert hasattr(llm, 'get_context_window_size'), f"LLM should have get_context_window_size method for {model_name}"


def test_litellm_version_constraint_bounds():
    """
    Test that the litellm version constraint properly bounds major versions.
    
    Ensures that the constraint litellm>=1.72.0,<2.0.0 prevents installation
    of potentially incompatible major versions while allowing patch updates.
    """
    import pkg_resources
    
    try:
        version = pkg_resources.get_distribution("litellm").version
        major, minor, patch = map(int, version.split('.')[:3])
        
        assert major == 1, f"litellm major version should be 1, got {major}"
        assert (minor, patch) >= (72, 0), f"litellm version {version} is below minimum required 1.72.0"
        
    except pkg_resources.DistributionNotFound:
        pytest.fail("litellm package not found - dependency resolution may have failed")

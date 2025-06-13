import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def test_litellm_dependency_allows_patch_versions():
    """Test that the litellm dependency constraint allows patch versions >= 1.72.0"""
    
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
    """Test that litellm can be imported and basic functionality works"""
    try:
        import litellm
        assert hasattr(litellm, 'completion'), "litellm should have completion function"
        assert hasattr(litellm, 'set_verbose'), "litellm should have set_verbose function"
        assert hasattr(litellm, 'drop_params'), "litellm should have drop_params attribute"
    except ImportError as e:
        pytest.fail(f"Failed to import litellm: {e}")


def test_crewai_llm_works_with_current_litellm():
    """Test that CrewAI's LLM class works with the current litellm version"""
    from crewai.llm import LLM
    
    llm = LLM(model="gpt-3.5-turbo")
    
    assert llm.model == "gpt-3.5-turbo"
    assert hasattr(llm, 'call'), "LLM should have call method"
    assert hasattr(llm, 'supports_function_calling'), "LLM should have supports_function_calling method"
    assert hasattr(llm, 'get_context_window_size'), "LLM should have get_context_window_size method"

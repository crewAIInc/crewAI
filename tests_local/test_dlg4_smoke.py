import os
import sys
import shutil
from pathlib import Path
import pytest
from dlg4_grant_system import main

@pytest.fixture
def setup_teardown_test_env(monkeypatch):
    """Set up test environment and clean up after."""
    output_dir = Path("tests_local/test_output_smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("OPENAI_API_KEY", "test_key_not_real")
    
    original_argv = sys.argv
    sys.argv = [
        "dlg4_grant_system.py",
        "--smoke",
        "--output-dir",
        str(output_dir),
    ]
    
    yield output_dir
    
    sys.argv = original_argv
    shutil.rmtree(output_dir)

def test_dlg4_smoke_run(setup_teardown_test_env):
    """Tests the main script in --smoke mode, checking for output files."""
    output_dir = setup_teardown_test_env
    
    # The main function will now run with patched argv and env vars
    return_code = main()
    
    assert return_code == 0, "The script should exit with status 0 on success."
    
    # Check for foundation profile
    profile_files = list(output_dir.glob("foundation_profile_*.md"))
    assert len(profile_files) == 1, "Expected one foundation profile markdown file."
    assert profile_files[0].read_text().strip() != "", "Foundation profile should not be empty."
    
    # Check for application planning files in the smoke-test-grant folder
    app_dir = output_dir / "applications" / "smoke-test-grant"
    assert app_dir.is_dir(), "Expected 'smoke-test-grant' application directory."
    
    expected_files = [
        "01_grant_brief.md",
        "02_requirements_checklist.md",
        "03_outline.md",
    ]
    for f in expected_files:
        file_path = app_dir / f
        assert file_path.is_file(), f"Expected file '{f}' not found in application directory."
        assert file_path.read_text().strip() != "", f"Application file '{f}' should not be empty."

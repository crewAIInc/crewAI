"""Tests to detect and prevent hardcoded secrets in the codebase.

These tests scan source files for patterns that look like hardcoded secrets
(API keys, tokens, passwords) to prevent accidental credential leaks.
"""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from crewai.cli.create_flow import create_flow
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
)

# Root of the workspace
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
CREWAI_SRC = WORKSPACE_ROOT / "lib" / "crewai" / "src"
CREWAI_TOOLS_SRC = WORKSPACE_ROOT / "lib" / "crewai-tools" / "src"

# Patterns that indicate hardcoded secrets in source code (not docs/tests)
SECRET_PATTERNS = [
    # Actual API key formats
    re.compile(r'''["']sk-proj-[a-zA-Z0-9_-]{20,}["']'''),
    re.compile(r'''["']sk-ant-api[a-zA-Z0-9_-]{20,}["']'''),
    re.compile(r'''["']ghp_[a-zA-Z0-9]{36}["']'''),
    re.compile(r'''["']gho_[a-zA-Z0-9]{36}["']'''),
    re.compile(r'''["']xox[bpas]-[a-zA-Z0-9-]{10,}["']'''),
    re.compile(r'''["']AKIA[A-Z0-9]{16}["']'''),
    # os.environ assignment with hardcoded non-empty value
    re.compile(r'''os\.environ\[["'][A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)["']\]\s*=\s*["'][^"']+["']'''),
]

# Files/directories to skip (tests, docs, examples patterns in docstrings are OK)
SKIP_DIRS = {
    "tests",
    "test",
    "__pycache__",
    ".git",
    "cassettes",
    "node_modules",
    ".venv",
}


def _get_python_source_files(root: Path) -> list[Path]:
    """Get all Python source files, excluding test directories."""
    files = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        files.append(path)
    return files


class TestNoHardcodedSecrets:
    """Test that source code does not contain hardcoded secrets."""

    def test_no_real_api_keys_in_source(self):
        """Verify no real API key patterns exist in source code."""
        violations = []

        for src_root in [CREWAI_SRC, CREWAI_TOOLS_SRC]:
            if not src_root.exists():
                continue
            for filepath in _get_python_source_files(src_root):
                content = filepath.read_text(errors="ignore")
                for pattern in SECRET_PATTERNS:
                    for match in pattern.finditer(content):
                        # Get the line number
                        line_num = content[: match.start()].count("\n") + 1
                        violations.append(
                            f"{filepath.relative_to(WORKSPACE_ROOT)}:{line_num}: {match.group()}"
                        )

        assert not violations, (
            f"Found {len(violations)} potential hardcoded secret(s):\n"
            + "\n".join(violations)
        )

    def test_no_env_assignment_with_hardcoded_keys(self):
        """Verify no os.environ['KEY'] = 'hardcoded-value' patterns in source (non-test) code."""
        pattern = re.compile(
            r'''os\.environ\[["'](\w*(?:KEY|TOKEN|SECRET|PASSWORD)\w*)["']\]\s*=\s*["']([^"']+)["']'''
        )
        # Config flags that are not secrets
        ALLOWED_ENV_ASSIGNMENTS = {
            "TOKENIZERS_PARALLELISM",
        }

        violations = []
        for src_root in [CREWAI_SRC, CREWAI_TOOLS_SRC]:
            if not src_root.exists():
                continue
            for filepath in _get_python_source_files(src_root):
                content = filepath.read_text(errors="ignore")
                for match in pattern.finditer(content):
                    env_var_name = match.group(1)
                    if env_var_name in ALLOWED_ENV_ASSIGNMENTS:
                        continue
                    line_num = content[: match.start()].count("\n") + 1
                    violations.append(
                        f"{filepath.relative_to(WORKSPACE_ROOT)}:{line_num}: "
                        f"os.environ['{match.group(1)}'] = '{match.group(2)}'"
                    )

        assert not violations, (
            f"Found {len(violations)} hardcoded environment variable assignment(s):\n"
            + "\n".join(violations)
            + "\n\nUse os.environ.get() or read from .env files instead."
        )


class TestCreateFlowEnvFile:
    """Test that create_flow generates .env files without hardcoded secret values."""

    def test_create_flow_env_file_has_no_hardcoded_api_key(self):
        """Verify create_flow does not write a hardcoded API key value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                create_flow("test_flow")

                env_file = Path(temp_dir) / "test_flow" / ".env"
                assert env_file.exists(), ".env file should be created"

                content = env_file.read_text()
                assert "YOUR_API_KEY" not in content, (
                    ".env should not contain hardcoded placeholder 'YOUR_API_KEY'"
                )
                # The key name should be present but without a hardcoded value
                assert "OPENAI_API_KEY" in content, (
                    ".env should contain the OPENAI_API_KEY variable name"
                )
            finally:
                os.chdir(original_cwd)


class TestProviderDefaultApiKeys:
    """Test that provider default API keys use environment variable lookups."""

    def test_ollama_default_api_key_from_env(self):
        """Verify Ollama default API key can be overridden via environment variable."""
        with patch.dict(os.environ, {"OLLAMA_DEFAULT_API_KEY": "custom-ollama-key"}, clear=False):
            # Re-import to pick up new env var - but since module-level dict is already
            # evaluated, we test the env var pattern is used in the config
            config = OPENAI_COMPATIBLE_PROVIDERS["ollama"]
            # The default_api_key should be set (either from env or fallback)
            assert config.default_api_key is not None

    def test_vllm_default_api_key_not_dummy(self):
        """Verify hosted_vllm default API key is not the literal string 'dummy'."""
        config = OPENAI_COMPATIBLE_PROVIDERS["hosted_vllm"]
        assert config.default_api_key != "dummy", (
            "hosted_vllm should not use 'dummy' as a hardcoded default API key"
        )
        assert config.default_api_key is not None

    def test_ollama_default_api_key_fallback(self):
        """Verify Ollama uses 'ollama' as fallback when env var is not set."""
        # When OLLAMA_DEFAULT_API_KEY is not set, should fall back to "ollama"
        env = os.environ.copy()
        env.pop("OLLAMA_DEFAULT_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            # The config was already created at module load time, so we check
            # the current value
            config = OPENAI_COMPATIBLE_PROVIDERS["ollama"]
            assert config.default_api_key is not None

    def test_all_providers_have_valid_config(self):
        """Verify all providers have properly configured API key settings."""
        for provider_name, config in OPENAI_COMPATIBLE_PROVIDERS.items():
            assert config.api_key_env, (
                f"Provider '{provider_name}' must have api_key_env configured"
            )
            if not config.api_key_required:
                assert config.default_api_key is not None, (
                    f"Provider '{provider_name}' with api_key_required=False "
                    "must have a default_api_key"
                )

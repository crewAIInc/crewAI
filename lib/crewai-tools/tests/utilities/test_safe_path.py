"""Tests for path and URL validation utilities."""

from __future__ import annotations

import os

import pytest

from crewai_tools.security.safe_path import (
    validate_directory_path,
    validate_file_path,
    validate_url,
)


# ---------------------------------------------------------------------------
# File path validation
# ---------------------------------------------------------------------------

class TestValidateFilePath:
    """Tests for validate_file_path."""

    def test_valid_relative_path(self, tmp_path):
        """Normal relative path within the base directory."""
        (tmp_path / "data.json").touch()
        result = validate_file_path("data.json", str(tmp_path))
        assert result == str(tmp_path / "data.json")

    def test_valid_nested_path(self, tmp_path):
        """Nested path within base directory."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "file.txt").touch()
        result = validate_file_path("sub/file.txt", str(tmp_path))
        assert result == str(tmp_path / "sub" / "file.txt")

    def test_rejects_dotdot_traversal(self, tmp_path):
        """Reject ../ traversal that escapes base_dir."""
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validate_file_path("../../etc/passwd", str(tmp_path))

    def test_rejects_absolute_path_outside_base(self, tmp_path):
        """Reject absolute path outside base_dir."""
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validate_file_path("/etc/passwd", str(tmp_path))

    def test_allows_absolute_path_inside_base(self, tmp_path):
        """Allow absolute path that's inside base_dir."""
        (tmp_path / "ok.txt").touch()
        result = validate_file_path(str(tmp_path / "ok.txt"), str(tmp_path))
        assert result == str(tmp_path / "ok.txt")

    def test_rejects_symlink_escape(self, tmp_path):
        """Reject symlinks that point outside base_dir."""
        link = tmp_path / "sneaky_link"
        # Create a symlink pointing to /etc/passwd
        os.symlink("/etc/passwd", str(link))
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validate_file_path("sneaky_link", str(tmp_path))

    def test_defaults_to_cwd(self):
        """When no base_dir is given, use cwd."""
        cwd = os.getcwd()
        # A file in cwd should be valid
        result = validate_file_path(".", None)
        assert result == os.path.realpath(cwd)

    def test_escape_hatch(self, tmp_path, monkeypatch):
        """CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true bypasses validation."""
        monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
        # This would normally be rejected
        result = validate_file_path("/etc/passwd", str(tmp_path))
        assert result == os.path.realpath("/etc/passwd")


class TestValidateDirectoryPath:
    """Tests for validate_directory_path."""

    def test_valid_directory(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = validate_directory_path("subdir", str(tmp_path))
        assert result == str(tmp_path / "subdir")

    def test_rejects_file_as_directory(self, tmp_path):
        (tmp_path / "file.txt").touch()
        with pytest.raises(ValueError, match="not a directory"):
            validate_directory_path("file.txt", str(tmp_path))

    def test_rejects_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="outside the allowed directory"):
            validate_directory_path("../../", str(tmp_path))


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

class TestValidateUrl:
    """Tests for validate_url."""

    def test_valid_https_url(self):
        """Normal HTTPS URL should pass."""
        result = validate_url("https://example.com/data.json")
        assert result == "https://example.com/data.json"

    def test_valid_http_url(self):
        """Normal HTTP URL should pass."""
        result = validate_url("http://example.com/api")
        assert result == "http://example.com/api"

    def test_blocks_file_scheme(self):
        """file:// URLs must be blocked."""
        with pytest.raises(ValueError, match="file:// URLs are not allowed"):
            validate_url("file:///etc/passwd")

    def test_blocks_file_scheme_with_host(self):
        with pytest.raises(ValueError, match="file:// URLs are not allowed"):
            validate_url("file://localhost/etc/shadow")

    def test_blocks_localhost(self):
        """localhost must be blocked (resolves to 127.0.0.1)."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://localhost/admin")

    def test_blocks_127_0_0_1(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://127.0.0.1/admin")

    def test_blocks_cloud_metadata(self):
        """AWS/GCP/Azure metadata endpoint must be blocked."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_private_10_range(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://10.0.0.1/internal")

    def test_blocks_private_172_range(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://172.16.0.1/internal")

    def test_blocks_private_192_range(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://192.168.1.1/router")

    def test_blocks_zero_address(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://0.0.0.0/")

    def test_blocks_ipv6_localhost(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://[::1]/admin")

    def test_blocks_ftp_scheme(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("ftp://example.com/file")

    def test_blocks_empty_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_url("http:///path")

    def test_blocks_unresolvable_host(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            validate_url("http://this-host-definitely-does-not-exist-abc123.com/")

    def test_escape_hatch(self, monkeypatch):
        """CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true bypasses URL validation."""
        monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
        # file:// would normally be blocked
        result = validate_url("file:///etc/passwd")
        assert result == "file:///etc/passwd"

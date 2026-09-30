"""Tests for path and URL validation utilities."""

from __future__ import annotations

import os

import pytest

from crewai_tools.security.safe_path import (
    format_path_for_display,
    format_sandbox_error,
    is_blocked_ip,
    validate_directory_path,
    validate_file_path,
    validate_url,
)


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

    def test_rejection_message_redacts_absolute_prefixes(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"

        with pytest.raises(ValueError) as exc_info:
            validate_file_path(str(outside), str(tmp_path))

        message = str(exc_info.value)
        assert "outside.txt" in message
        assert str(tmp_path) not in message
        assert str(tmp_path.parent) not in message


class TestFormatPathForDisplay:
    """Tests for user-visible path labels."""

    def test_returns_relative_path_inside_base(self, tmp_path):
        nested_file = tmp_path / "nested" / "file.txt"
        nested_file.parent.mkdir()
        nested_file.touch()

        result = format_path_for_display(str(nested_file), str(tmp_path))

        assert result == os.path.join("nested", "file.txt")

    def test_redacts_absolute_prefix_outside_base(self, tmp_path):
        outside_file = tmp_path.parent / "outside.txt"

        result = format_path_for_display(str(outside_file), str(tmp_path))

        assert result == "outside.txt"


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

    def test_force_safe_paths_overrides_escape_hatch(self, monkeypatch):
        """Managed workers can ignore a tenant-supplied escape hatch."""
        monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
        monkeypatch.setenv("CREWAI_TOOLS_FORCE_SAFE_PATHS", "true")
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://127.0.0.1/admin")


class TestIsBlockedIp:
    """Tests for is_blocked_ip."""

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "0.0.0.0",
            "::1",
            "::",
            "fc00::1",
            "fe80::1",
            "169.254.169.254",  # AWS/GCP/Azure metadata
            "168.63.129.16",  # Azure WireServer
            "fd00:ec2::254",  # AWS IMDS over IPv6
            "100.64.0.1",  # CGNAT start
            "100.100.100.200",  # Alibaba metadata
            "100.127.255.255",  # CGNAT end
            "198.18.0.1",  # benchmarking start
            "198.19.255.255",  # benchmarking end
            "64:ff9b::a00:1",  # NAT64 of 10.0.0.1
            "64:ff9b:1::a00:1",  # local-use NAT64
            "2002:c0a8:0101::1",  # 6to4 of 192.168.1.1
            "::ffff:100.100.100.200",  # IPv4-mapped
            "::ffff:127.0.0.1",
        ],
    )
    def test_blocks_non_public_addresses(self, ip):
        assert is_blocked_ip(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
            "100.63.255.255",  # just below CGNAT
            "100.128.0.0",  # just above CGNAT
            "198.17.255.255",
            "198.20.0.0",
            "64:ff9b::808:808",  # NAT64 of public 8.8.8.8
            "::ffff:8.8.8.8",
        ],
    )
    def test_allows_public_addresses(self, ip):
        assert is_blocked_ip(ip) is False

    @pytest.mark.parametrize("ip", ["", "not-an-ip", "999.1.1.1", "1.2.3"])
    def test_blocks_unparseable_input(self, ip):
        assert is_blocked_ip(ip) is True


class TestFormatSandboxError:
    def test_replaces_bypass_advice_with_remedy(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            validate_file_path(str(tmp_path.parent / "outside.txt"), str(tmp_path))

        message = format_sandbox_error(exc.value, "Pass base_dir to widen it.")

        assert "outside the allowed directory" in message
        assert "Pass base_dir to widen it." in message
        assert "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS" not in message

    def test_leaves_unrelated_errors_intact(self):
        message = format_sandbox_error(ValueError("something else"), "Do this.")

        assert message == "something else Do this."

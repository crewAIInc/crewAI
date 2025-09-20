"""Tests for the machine ID generation functionality in tracing utils."""

from pathlib import Path
from unittest.mock import patch

from crewai.events.listeners.tracing.utils import (
    _get_generic_system_id,
    _get_linux_machine_id,
    _get_machine_id,
)


def test_get_machine_id_basic():
    """Test that _get_machine_id always returns a valid SHA256 hash."""
    machine_id = _get_machine_id()

    # Should return a 64-character hex string (SHA256)
    assert isinstance(machine_id, str)
    assert len(machine_id) == 64
    assert all(c in "0123456789abcdef" for c in machine_id)


def test_get_machine_id_handles_missing_files():
    """Test that _get_machine_id handles FileNotFoundError gracefully."""
    with patch.object(Path, "read_text", side_effect=FileNotFoundError):
        machine_id = _get_machine_id()

        # Should still return a valid hash even when files are missing
        assert isinstance(machine_id, str)
        assert len(machine_id) == 64
        assert all(c in "0123456789abcdef" for c in machine_id)


def test_get_machine_id_handles_permission_errors():
    """Test that _get_machine_id handles PermissionError gracefully."""
    with patch.object(Path, "read_text", side_effect=PermissionError):
        machine_id = _get_machine_id()

        # Should still return a valid hash even with permission errors
        assert isinstance(machine_id, str)
        assert len(machine_id) == 64
        assert all(c in "0123456789abcdef" for c in machine_id)


def test_get_machine_id_handles_mac_address_failure():
    """Test that _get_machine_id works even if MAC address retrieval fails."""
    with patch("uuid.getnode", side_effect=Exception("MAC address error")):
        machine_id = _get_machine_id()

        # Should still return a valid hash even without MAC address
        assert isinstance(machine_id, str)
        assert len(machine_id) == 64
        assert all(c in "0123456789abcdef" for c in machine_id)


def test_get_linux_machine_id_handles_missing_files():
    """Test that _get_linux_machine_id handles missing files gracefully."""
    with patch.object(Path, "exists", return_value=False):
        result = _get_linux_machine_id()

        # Should return something (hostname-arch fallback) or None
        assert result is None or isinstance(result, str)


def test_get_linux_machine_id_handles_file_read_errors():
    """Test that _get_linux_machine_id handles file read errors."""
    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "is_file", return_value=True),
        patch.object(Path, "read_text", side_effect=FileNotFoundError),
    ):
        result = _get_linux_machine_id()

        # Should fallback to hostname-based ID or None
        assert result is None or isinstance(result, str)


def test_get_generic_system_id_basic():
    """Test that _get_generic_system_id returns reasonable values."""
    result = _get_generic_system_id()

    # Should return a string or None
    assert result is None or isinstance(result, str)

    # If it returns a string, it should be non-empty
    if result:
        assert len(result) > 0


def test_get_generic_system_id_handles_socket_errors():
    """Test that _get_generic_system_id handles socket errors gracefully."""
    with patch("socket.gethostname", side_effect=Exception("Socket error")):
        result = _get_generic_system_id()

        # Should still work or return None
        assert result is None or isinstance(result, str)


def test_machine_id_consistency():
    """Test that machine ID is consistent across multiple calls."""
    machine_id1 = _get_machine_id()
    machine_id2 = _get_machine_id()

    # Should be the same across calls (stable fingerprint)
    assert machine_id1 == machine_id2


def test_machine_id_always_has_fallback():
    """Test that machine ID always generates something even in worst case."""
    with (
        patch("uuid.getnode", side_effect=Exception),
        patch("platform.system", side_effect=Exception),
        patch("socket.gethostname", side_effect=Exception),
        patch("getpass.getuser", side_effect=Exception),
        patch("platform.machine", side_effect=Exception),
        patch("platform.processor", side_effect=Exception),
        patch.object(Path, "read_text", side_effect=FileNotFoundError),
    ):
        machine_id = _get_machine_id()

        # Even in worst case, should return a valid hash
        assert isinstance(machine_id, str)
        assert len(machine_id) == 64
        assert all(c in "0123456789abcdef" for c in machine_id)

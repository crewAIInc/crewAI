"""Test for version management."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai import __version__
from crewai.cli.version import (
    _get_cache_file,
    _is_cache_valid,
    get_crewai_version,
    get_latest_version_from_pypi,
    is_newer_version_available,
)


def test_dynamic_versioning_consistency() -> None:
    """Test that dynamic versioning provides consistent version across all access methods."""
    cli_version = get_crewai_version()
    package_version = __version__

    # Both should return the same version string
    assert cli_version == package_version

    # Version should not be empty
    assert package_version is not None
    assert len(package_version.strip()) > 0


class TestVersionChecking:
    """Test version checking utilities."""

    def test_get_crewai_version(self) -> None:
        """Test getting current crewai version."""
        version = get_crewai_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_cache_file(self) -> None:
        """Test cache file path generation."""
        cache_file = _get_cache_file()
        assert isinstance(cache_file, Path)
        assert cache_file.name == "version_cache.json"

    def test_is_cache_valid_with_fresh_cache(self) -> None:
        """Test cache validation with fresh cache."""
        cache_data = {"timestamp": datetime.now().isoformat(), "version": "1.0.0"}
        assert _is_cache_valid(cache_data) is True

    def test_is_cache_valid_with_stale_cache(self) -> None:
        """Test cache validation with stale cache."""
        old_time = datetime.now() - timedelta(hours=25)
        cache_data = {"timestamp": old_time.isoformat(), "version": "1.0.0"}
        assert _is_cache_valid(cache_data) is False

    def test_is_cache_valid_with_missing_timestamp(self) -> None:
        """Test cache validation with missing timestamp."""
        cache_data = {"version": "1.0.0"}
        assert _is_cache_valid(cache_data) is False

    @patch("crewai.cli.version.Path.exists")
    @patch("crewai.cli.version.request.urlopen")
    def test_get_latest_version_from_pypi_success(
        self, mock_urlopen: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test successful PyPI version fetch."""
        # Mock cache not existing to force fetch from PyPI
        mock_exists.return_value = False

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"info": {"version": "2.0.0"}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        version = get_latest_version_from_pypi()
        assert version == "2.0.0"

    @patch("crewai.cli.version.Path.exists")
    @patch("crewai.cli.version.request.urlopen")
    def test_get_latest_version_from_pypi_failure(
        self, mock_urlopen: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test PyPI version fetch failure."""
        from urllib.error import URLError

        # Mock cache not existing to force fetch from PyPI
        mock_exists.return_value = False

        mock_urlopen.side_effect = URLError("Network error")

        version = get_latest_version_from_pypi()
        assert version is None

    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version.get_latest_version_from_pypi")
    def test_is_newer_version_available_true(
        self, mock_latest: MagicMock, mock_current: MagicMock
    ) -> None:
        """Test when newer version is available."""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = "2.0.0"

        is_newer, current, latest = is_newer_version_available()
        assert is_newer is True
        assert current == "1.0.0"
        assert latest == "2.0.0"

    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version.get_latest_version_from_pypi")
    def test_is_newer_version_available_false(
        self, mock_latest: MagicMock, mock_current: MagicMock
    ) -> None:
        """Test when no newer version is available."""
        mock_current.return_value = "2.0.0"
        mock_latest.return_value = "2.0.0"

        is_newer, current, latest = is_newer_version_available()
        assert is_newer is False
        assert current == "2.0.0"
        assert latest == "2.0.0"

    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version.get_latest_version_from_pypi")
    def test_is_newer_version_available_with_none_latest(
        self, mock_latest: MagicMock, mock_current: MagicMock
    ) -> None:
        """Test when PyPI fetch fails."""
        mock_current.return_value = "1.0.0"
        mock_latest.return_value = None

        is_newer, current, latest = is_newer_version_available()
        assert is_newer is False
        assert current == "1.0.0"
        assert latest is None


class TestConsoleFormatterVersionCheck:
    """Test version check display in ConsoleFormatter."""

    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": ""})
    def test_version_message_shows_when_update_available_and_verbose(
        self, mock_check: MagicMock
    ) -> None:
        """Test version message shows when update available and verbose enabled."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            assert mock_print.call_count == 2

    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    def test_version_message_hides_when_verbose_false(
        self, mock_check: MagicMock
    ) -> None:
        """Test version message hidden when verbose disabled."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")

        formatter = ConsoleFormatter(verbose=False)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            mock_print.assert_not_called()

    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    def test_version_message_hides_when_no_update_available(
        self, mock_check: MagicMock
    ) -> None:
        """Test version message hidden when no update available."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (False, "2.0.0", "2.0.0")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            mock_print.assert_not_called()

    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": "true"})
    def test_version_message_hides_in_ci_environment(
        self, mock_check: MagicMock
    ) -> None:
        """Test version message hidden when running in CI/CD."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            mock_print.assert_not_called()

    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": "1"})
    def test_version_message_hides_in_ci_environment_with_numeric_value(
        self, mock_check: MagicMock
    ) -> None:
        """Test version message hidden when CI=1."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            mock_print.assert_not_called()

"""Test for version management."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai import __version__
from crewai.cli.version import (
    _find_latest_non_yanked_version,
    _get_cache_file,
    _is_cache_valid,
    _is_version_yanked,
    get_crewai_version,
    get_latest_version_from_pypi,
    is_current_version_yanked,
    is_newer_version_available,
)


def test_dynamic_versioning_consistency() -> None:
    """Test that dynamic versioning provides consistent version across all access methods."""
    cli_version = get_crewai_version()
    package_version = __version__

    assert cli_version == package_version

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
        """Test successful PyPI version fetch uses releases data."""
        mock_exists.return_value = False

        releases = {
            "1.0.0": [{"yanked": False}],
            "2.0.0": [{"yanked": False}],
            "2.1.0": [{"yanked": True, "yanked_reason": "bad release"}],
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"info": {"version": "2.1.0"}, "releases": releases}
        ).encode()
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


class TestFindLatestNonYankedVersion:
    """Test _find_latest_non_yanked_version helper."""

    def test_skips_yanked_versions(self) -> None:
        """Test that yanked versions are skipped."""
        releases = {
            "1.0.0": [{"yanked": False}],
            "2.0.0": [{"yanked": True}],
        }
        assert _find_latest_non_yanked_version(releases) == "1.0.0"

    def test_returns_highest_non_yanked(self) -> None:
        """Test that the highest non-yanked version is returned."""
        releases = {
            "1.0.0": [{"yanked": False}],
            "1.5.0": [{"yanked": False}],
            "2.0.0": [{"yanked": True}],
        }
        assert _find_latest_non_yanked_version(releases) == "1.5.0"

    def test_returns_none_when_all_yanked(self) -> None:
        """Test that None is returned when all versions are yanked."""
        releases = {
            "1.0.0": [{"yanked": True}],
            "2.0.0": [{"yanked": True}],
        }
        assert _find_latest_non_yanked_version(releases) is None

    def test_skips_prerelease_versions(self) -> None:
        """Test that pre-release versions are skipped."""
        releases = {
            "1.0.0": [{"yanked": False}],
            "2.0.0a1": [{"yanked": False}],
            "2.0.0rc1": [{"yanked": False}],
        }
        assert _find_latest_non_yanked_version(releases) == "1.0.0"

    def test_skips_versions_with_empty_files(self) -> None:
        """Test that versions with no files are skipped."""
        releases: dict[str, list[dict[str, bool]]] = {
            "1.0.0": [{"yanked": False}],
            "2.0.0": [],
        }
        assert _find_latest_non_yanked_version(releases) == "1.0.0"

    def test_handles_invalid_version_strings(self) -> None:
        """Test that invalid version strings are skipped."""
        releases = {
            "1.0.0": [{"yanked": False}],
            "not-a-version": [{"yanked": False}],
        }
        assert _find_latest_non_yanked_version(releases) == "1.0.0"

    def test_partially_yanked_files_not_considered_yanked(self) -> None:
        """Test that a version with some non-yanked files is not yanked."""
        releases = {
            "1.0.0": [{"yanked": False}],
            "2.0.0": [{"yanked": True}, {"yanked": False}],
        }
        assert _find_latest_non_yanked_version(releases) == "2.0.0"


class TestIsVersionYanked:
    """Test _is_version_yanked helper."""

    def test_non_yanked_version(self) -> None:
        """Test a non-yanked version returns False."""
        releases = {"1.0.0": [{"yanked": False}]}
        is_yanked, reason = _is_version_yanked("1.0.0", releases)
        assert is_yanked is False
        assert reason == ""

    def test_yanked_version_with_reason(self) -> None:
        """Test a yanked version returns True with reason."""
        releases = {
            "1.0.0": [{"yanked": True, "yanked_reason": "critical bug"}],
        }
        is_yanked, reason = _is_version_yanked("1.0.0", releases)
        assert is_yanked is True
        assert reason == "critical bug"

    def test_yanked_version_without_reason(self) -> None:
        """Test a yanked version returns True with empty reason."""
        releases = {"1.0.0": [{"yanked": True}]}
        is_yanked, reason = _is_version_yanked("1.0.0", releases)
        assert is_yanked is True
        assert reason == ""

    def test_unknown_version(self) -> None:
        """Test an unknown version returns False."""
        releases = {"1.0.0": [{"yanked": False}]}
        is_yanked, reason = _is_version_yanked("9.9.9", releases)
        assert is_yanked is False
        assert reason == ""

    def test_partially_yanked_files(self) -> None:
        """Test a version with mixed yanked/non-yanked files is not yanked."""
        releases = {
            "1.0.0": [{"yanked": True}, {"yanked": False}],
        }
        is_yanked, reason = _is_version_yanked("1.0.0", releases)
        assert is_yanked is False
        assert reason == ""

    def test_multiple_yanked_files_picks_first_reason(self) -> None:
        """Test that the first available reason is returned."""
        releases = {
            "1.0.0": [
                {"yanked": True, "yanked_reason": ""},
                {"yanked": True, "yanked_reason": "second reason"},
            ],
        }
        is_yanked, reason = _is_version_yanked("1.0.0", releases)
        assert is_yanked is True
        assert reason == "second reason"


class TestIsCurrentVersionYanked:
    """Test is_current_version_yanked public function."""

    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version._get_cache_file")
    def test_reads_from_valid_cache(
        self, mock_cache_file: MagicMock, mock_version: MagicMock, tmp_path: Path
    ) -> None:
        """Test reading yanked status from a valid cache."""
        mock_version.return_value = "1.0.0"
        cache_file = tmp_path / "version_cache.json"
        cache_data = {
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "current_version": "1.0.0",
            "current_version_yanked": True,
            "current_version_yanked_reason": "bad release",
        }
        cache_file.write_text(json.dumps(cache_data))
        mock_cache_file.return_value = cache_file

        is_yanked, reason = is_current_version_yanked()
        assert is_yanked is True
        assert reason == "bad release"

    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version._get_cache_file")
    def test_not_yanked_from_cache(
        self, mock_cache_file: MagicMock, mock_version: MagicMock, tmp_path: Path
    ) -> None:
        """Test non-yanked status from a valid cache."""
        mock_version.return_value = "2.0.0"
        cache_file = tmp_path / "version_cache.json"
        cache_data = {
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "current_version": "2.0.0",
            "current_version_yanked": False,
            "current_version_yanked_reason": "",
        }
        cache_file.write_text(json.dumps(cache_data))
        mock_cache_file.return_value = cache_file

        is_yanked, reason = is_current_version_yanked()
        assert is_yanked is False
        assert reason == ""

    @patch("crewai.cli.version.get_latest_version_from_pypi")
    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version._get_cache_file")
    def test_triggers_fetch_on_stale_cache(
        self,
        mock_cache_file: MagicMock,
        mock_version: MagicMock,
        mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that a stale cache triggers a re-fetch."""
        mock_version.return_value = "1.0.0"
        cache_file = tmp_path / "version_cache.json"
        old_time = datetime.now() - timedelta(hours=25)
        cache_data = {
            "version": "2.0.0",
            "timestamp": old_time.isoformat(),
            "current_version": "1.0.0",
            "current_version_yanked": True,
            "current_version_yanked_reason": "old reason",
        }
        cache_file.write_text(json.dumps(cache_data))
        mock_cache_file.return_value = cache_file

        fresh_cache = {
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "current_version": "1.0.0",
            "current_version_yanked": False,
            "current_version_yanked_reason": "",
        }

        def write_fresh_cache() -> str:
            cache_file.write_text(json.dumps(fresh_cache))
            return "2.0.0"

        mock_fetch.side_effect = lambda: write_fresh_cache()

        is_yanked, reason = is_current_version_yanked()
        assert is_yanked is False
        mock_fetch.assert_called_once()

    @patch("crewai.cli.version.get_latest_version_from_pypi")
    @patch("crewai.cli.version.get_crewai_version")
    @patch("crewai.cli.version._get_cache_file")
    def test_returns_false_on_fetch_failure(
        self,
        mock_cache_file: MagicMock,
        mock_version: MagicMock,
        mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that fetch failure returns not yanked."""
        mock_version.return_value = "1.0.0"
        cache_file = tmp_path / "version_cache.json"
        mock_cache_file.return_value = cache_file
        mock_fetch.return_value = None

        is_yanked, reason = is_current_version_yanked()
        assert is_yanked is False
        assert reason == ""


class TestConsoleFormatterVersionCheck:
    """Test version check display in ConsoleFormatter."""

    @patch("crewai.events.utils.console_formatter.is_current_version_yanked")
    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": ""})
    def test_version_message_shows_when_update_available_and_verbose(
        self, mock_check: MagicMock, mock_yanked: MagicMock
    ) -> None:
        """Test version message shows when update available and verbose enabled."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")
        mock_yanked.return_value = (False, "")

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

    @patch("crewai.events.utils.console_formatter.is_current_version_yanked")
    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    def test_version_message_hides_when_no_update_available(
        self, mock_check: MagicMock, mock_yanked: MagicMock
    ) -> None:
        """Test version message hidden when no update available."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (False, "2.0.0", "2.0.0")
        mock_yanked.return_value = (False, "")

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

    @patch("crewai.events.utils.console_formatter.is_current_version_yanked")
    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": ""})
    def test_yanked_warning_shows_when_version_is_yanked(
        self, mock_check: MagicMock, mock_yanked: MagicMock
    ) -> None:
        """Test yanked warning panel shows when current version is yanked."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (False, "1.0.0", "1.0.0")
        mock_yanked.return_value = (True, "critical bug")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            assert mock_print.call_count == 2
            panel = mock_print.call_args_list[0][0][0]
            assert "Yanked Version" in panel.title
            assert "critical bug" in str(panel.renderable)

    @patch("crewai.events.utils.console_formatter.is_current_version_yanked")
    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": ""})
    def test_yanked_warning_shows_without_reason(
        self, mock_check: MagicMock, mock_yanked: MagicMock
    ) -> None:
        """Test yanked warning panel shows even without a reason."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (False, "1.0.0", "1.0.0")
        mock_yanked.return_value = (True, "")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            assert mock_print.call_count == 2
            panel = mock_print.call_args_list[0][0][0]
            assert "Yanked Version" in panel.title
            assert "Reason:" not in str(panel.renderable)

    @patch("crewai.events.utils.console_formatter.is_current_version_yanked")
    @patch("crewai.events.utils.console_formatter.is_newer_version_available")
    @patch.dict("os.environ", {"CI": ""})
    def test_both_update_and_yanked_warning_show(
        self, mock_check: MagicMock, mock_yanked: MagicMock
    ) -> None:
        """Test both update and yanked panels show when applicable."""
        from crewai.events.utils.console_formatter import ConsoleFormatter

        mock_check.return_value = (True, "1.0.0", "2.0.0")
        mock_yanked.return_value = (True, "security issue")

        formatter = ConsoleFormatter(verbose=True)
        with patch.object(formatter.console, "print") as mock_print:
            formatter._show_version_update_message_if_needed()
            assert mock_print.call_count == 4

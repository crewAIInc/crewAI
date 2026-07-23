"""Tests for PyPI polling URL validation in the release CLI."""

from unittest.mock import MagicMock, patch

from crewai_devtools.cli import _wait_for_pypi
import pytest


@patch("crewai_devtools.cli.requests.get")
@patch(
    "crewai_devtools.cli.validate_url",
    return_value="https://pypi.org/pypi/crewai/1.0.0/json",
)
def test_wait_for_pypi_validates_url_before_request(mock_validate_url, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    _wait_for_pypi("crewai", "1.0.0")

    mock_validate_url.assert_called_once_with("https://pypi.org/pypi/crewai/1.0.0/json")
    mock_get.assert_called_once_with(
        "https://pypi.org/pypi/crewai/1.0.0/json", timeout=30
    )


@patch("crewai_devtools.cli.requests.get")
@patch(
    "crewai_devtools.cli.validate_url",
    side_effect=ValueError("URL resolves to private/reserved IP"),
)
def test_wait_for_pypi_rejects_unsafe_url(mock_validate_url, mock_get):
    with pytest.raises(ValueError, match="private/reserved IP"):
        _wait_for_pypi("crewai", "1.0.0")

    mock_validate_url.assert_called_once()
    mock_get.assert_not_called()

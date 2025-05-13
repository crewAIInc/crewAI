import tempfile
from pathlib import Path
from unittest import mock

import pytest
import requests

from crewai.cli.provider import (
    fetch_provider_data,
    get_provider_data,
    load_provider_data,
)


@pytest.fixture
def mock_response():
    """Mock a successful response from requests.get."""
    mock_resp = mock.Mock()
    mock_resp.headers = {"content-length": "100"}
    mock_resp.iter_content.return_value = [b'{"model1": {"litellm_provider": "openai"}}']
    return mock_resp


@pytest.fixture
def mock_cache_file():
    """Create a temporary file to use as a cache file."""
    with tempfile.NamedTemporaryFile() as tmp:
        yield Path(tmp.name)


class TestProviderFunctions:
    @mock.patch("crewai.cli.provider.requests.get")
    def test_fetch_provider_data_with_ssl_verify(self, mock_get, mock_response, mock_cache_file):
        """Test that fetch_provider_data calls requests.get with verify=True by default."""
        mock_get.return_value = mock_response

        fetch_provider_data(mock_cache_file)

        mock_get.assert_called_once()
        assert mock_get.call_args[1]["verify"] is True

    @mock.patch("crewai.cli.provider.requests.get")
    def test_fetch_provider_data_without_ssl_verify(self, mock_get, mock_response, mock_cache_file):
        """Test that fetch_provider_data calls requests.get with verify=False when skip_ssl_verify=True."""
        mock_get.return_value = mock_response

        fetch_provider_data(mock_cache_file, skip_ssl_verify=True)

        mock_get.assert_called_once()
        assert mock_get.call_args[1]["verify"] is False

    @mock.patch("crewai.cli.provider.requests.get")
    def test_fetch_provider_data_handles_request_exception(self, mock_get, mock_cache_file):
        """Test that fetch_provider_data handles RequestException properly."""
        mock_get.side_effect = requests.RequestException("Test error")

        result = fetch_provider_data(mock_cache_file)

        assert result is None
        mock_get.assert_called_once()

    @mock.patch("crewai.cli.provider.fetch_provider_data")
    @mock.patch("crewai.cli.provider.read_cache_file")
    def test_load_provider_data_with_ssl_verify(
        self, mock_read_cache, mock_fetch, mock_cache_file
    ):
        """Test that load_provider_data passes skip_ssl_verify to fetch_provider_data."""
        mock_read_cache.return_value = None
        mock_fetch.return_value = {"model1": {"litellm_provider": "openai"}}

        load_provider_data(mock_cache_file, 3600, skip_ssl_verify=True)

        mock_fetch.assert_called_once_with(mock_cache_file, True)

    @mock.patch("crewai.cli.provider.load_provider_data")
    def test_get_provider_data_with_ssl_verify(self, mock_load, tmp_path):
        """Test that get_provider_data passes skip_ssl_verify to load_provider_data."""
        mock_load.return_value = {"model1": {"litellm_provider": "openai"}}

        get_provider_data(skip_ssl_verify=True)

        mock_load.assert_called_once()
        assert mock_load.call_args[0][2] is True  # skip_ssl_verify parameter

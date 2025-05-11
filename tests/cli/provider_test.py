import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import requests
from requests.exceptions import SSLError

from crewai.cli.provider import fetch_provider_data, get_provider_data


class TestProviderFunctions:
    @mock.patch("crewai.cli.provider.requests.get")
    def test_fetch_provider_data_success(self, mock_get):
        mock_response = mock.MagicMock()
        mock_response.headers.get.return_value = "100"
        mock_response.iter_content.return_value = [b'{"test": "data"}']
        mock_get.return_value = mock_response

        with tempfile.NamedTemporaryFile() as temp_file:
            cache_file = Path(temp_file.name)
            result = fetch_provider_data(cache_file)

        assert result == {"test": "data"}
        mock_get.assert_called_once()

    @mock.patch("crewai.cli.provider.requests.get")
    @mock.patch("crewai.cli.provider.click.secho")
    def test_fetch_provider_data_ssl_error_fallback(self, mock_secho, mock_get):
        mock_response = mock.MagicMock()
        mock_response.headers.get.return_value = "100"
        mock_response.iter_content.return_value = [b'{"test": "data"}']
        
        mock_get.side_effect = [
            SSLError("certificate verify failed: unable to get local issuer certificate"),
            mock_response
        ]

        with tempfile.NamedTemporaryFile() as temp_file:
            cache_file = Path(temp_file.name)
            result = fetch_provider_data(cache_file)

        assert result == {"test": "data"}
        assert mock_get.call_count == 2
        
        
        assert mock_get.call_args_list[1][1]["verify"] is False
        
        mock_secho.assert_any_call(
            "SSL certificate verification failed. Retrying with verification disabled. "
            "This is less secure but may be necessary on some systems.",
            fg="yellow"
        )

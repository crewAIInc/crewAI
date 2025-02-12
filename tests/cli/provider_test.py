from unittest.mock import Mock, patch

import json
import os
import pytest
import requests
import time

from crewai.cli.constants import JSON_URL, MODELS, PROVIDERS
from crewai.cli.provider import fetch_provider_data, get_provider_data

def test_fetch_provider_data_timeout():
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout
        result = fetch_provider_data('/tmp/cache.json')
        assert result is None

def test_fetch_provider_data_wrong_content_type():
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/plain'}
        mock_get.return_value = mock_response
        result = fetch_provider_data('/tmp/cache.json')
        assert result is None

def test_fetch_provider_data_success():
    mock_data = {"model1": {"provider": "test"}}
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = mock_data
        mock_response.iter_content.return_value = [json.dumps(mock_data).encode()]
        mock_get.return_value = mock_response
        result = fetch_provider_data('/tmp/cache.json')
        assert result == mock_data

def test_cache_expiry():
    with patch('os.path.getmtime') as mock_time:
        mock_time.return_value = time.time() - (25 * 60 * 60)  # 25 hours old
        with patch('crewai.cli.provider.load_provider_data') as mock_load:
            mock_load.return_value = None
            result = get_provider_data()
            assert result is not None
            assert all(provider.lower() in result for provider in PROVIDERS)
            # Verify that each provider has its models from MODELS
            for provider in PROVIDERS:
                assert result[provider.lower()] == MODELS.get(provider.lower(), [])

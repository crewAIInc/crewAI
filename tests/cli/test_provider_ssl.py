import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from crewai.cli.provider import fetch_provider_data, get_ssl_verify_config


class TestSSLConfiguration:
    def test_get_ssl_verify_config_with_requests_ca_bundle(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch.dict(os.environ, {'REQUESTS_CA_BUNDLE': temp_path}):
                result = get_ssl_verify_config()
                assert result == temp_path
        finally:
            os.unlink(temp_path)

    def test_get_ssl_verify_config_with_ssl_cert_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch.dict(os.environ, {'SSL_CERT_FILE': temp_path}, clear=True):
                result = get_ssl_verify_config()
                assert result == temp_path
        finally:
            os.unlink(temp_path)

    def test_get_ssl_verify_config_with_curl_ca_bundle(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with patch.dict(os.environ, {'CURL_CA_BUNDLE': temp_path}, clear=True):
                result = get_ssl_verify_config()
                assert result == temp_path
        finally:
            os.unlink(temp_path)

    def test_get_ssl_verify_config_precedence(self):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
            temp_path1 = temp_file1.name
        with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
            temp_path2 = temp_file2.name
        
        try:
            with patch.dict(os.environ, {
                'REQUESTS_CA_BUNDLE': temp_path1,
                'SSL_CERT_FILE': temp_path2
            }):
                result = get_ssl_verify_config()
                assert result == temp_path1
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)

    def test_get_ssl_verify_config_invalid_file(self):
        with patch.dict(os.environ, {'REQUESTS_CA_BUNDLE': '/nonexistent/file'}, clear=True):
            with patch('certifi.where', return_value='/path/to/certifi/cacert.pem'):
                result = get_ssl_verify_config()
                assert result == '/path/to/certifi/cacert.pem'

    def test_get_ssl_verify_config_fallback_to_certifi(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('certifi.where', return_value='/path/to/certifi/cacert.pem'):
                result = get_ssl_verify_config()
                assert result == '/path/to/certifi/cacert.pem'


class TestFetchProviderDataSSL:
    def test_fetch_provider_data_uses_ssl_config(self):
        cache_file = Path("/tmp/test_cache.json")
        mock_response = Mock()
        mock_response.headers = {'content-length': '100'}
        mock_response.iter_content.return_value = [b'{"test": "data"}']
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            with patch('crewai.cli.provider.get_ssl_verify_config', return_value='/custom/ca/bundle.pem'):
                fetch_provider_data(cache_file)
                
                mock_get.assert_called_once()
                args, kwargs = mock_get.call_args
                assert kwargs['verify'] == '/custom/ca/bundle.pem'
        
        if cache_file.exists():
            cache_file.unlink()

    def test_fetch_provider_data_ssl_error_handling(self):
        cache_file = Path("/tmp/test_cache.json")
        
        with patch('requests.get', side_effect=requests.exceptions.SSLError("SSL verification failed")):
            with patch('click.secho') as mock_secho:
                result = fetch_provider_data(cache_file)
                
                assert result is None
                mock_secho.assert_any_call("SSL certificate verification failed: SSL verification failed", fg="red")
                mock_secho.assert_any_call("Try setting REQUESTS_CA_BUNDLE environment variable to your CA bundle path", fg="yellow")

    def test_fetch_provider_data_general_request_error(self):
        cache_file = Path("/tmp/test_cache.json")
        
        with patch('requests.get', side_effect=requests.exceptions.RequestException("Network error")):
            with patch('click.secho') as mock_secho:
                result = fetch_provider_data(cache_file)
                
                assert result is None
                mock_secho.assert_any_call("Error fetching provider data: Network error", fg="red")

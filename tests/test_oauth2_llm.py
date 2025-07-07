import json
import pytest
import requests
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from crewai.llm import LLM
from crewai.llms.oauth2_config import OAuth2Config, OAuth2ConfigLoader
from crewai.llms.oauth2_token_manager import OAuth2TokenManager


class TestOAuth2Config:
    def test_oauth2_config_creation(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider",
            scope="read write"
        )
        assert config.client_id == "test_client"
        assert config.provider_name == "test_provider"
        assert config.scope == "read write"

    def test_oauth2_config_loader_missing_file(self):
        loader = OAuth2ConfigLoader("nonexistent.json")
        configs = loader.load_config()
        assert configs == {}

    def test_oauth2_config_loader_valid_config(self):
        config_data = {
            "oauth2_providers": {
                "custom_provider": {
                    "client_id": "test_client",
                    "client_secret": "test_secret", 
                    "token_url": "https://example.com/token",
                    "scope": "read"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            loader = OAuth2ConfigLoader(config_path)
            configs = loader.load_config()
            
            assert "custom_provider" in configs
            config = configs["custom_provider"]
            assert config.client_id == "test_client"
            assert config.provider_name == "custom_provider"
        finally:
            Path(config_path).unlink()

    def test_oauth2_config_loader_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json")
            config_path = f.name
        
        try:
            loader = OAuth2ConfigLoader(config_path)
            with pytest.raises(ValueError, match="Invalid OAuth2 configuration"):
                loader.load_config()
        finally:
            Path(config_path).unlink()


class TestOAuth2TokenManager:
    def test_token_acquisition_success(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "test_token_123",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            manager = OAuth2TokenManager()
            token = manager.get_access_token(config)
            
            assert token == "test_token_123"
            mock_post.assert_called_once()

    def test_token_caching(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "test_token_123",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            manager = OAuth2TokenManager()
            
            token1 = manager.get_access_token(config)
            assert token1 == "test_token_123"
            assert mock_post.call_count == 1
            
            token2 = manager.get_access_token(config)
            assert token2 == "test_token_123"
            assert mock_post.call_count == 1

    def test_token_refresh_on_expiry(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_token_456",
            "token_type": "Bearer", 
            "expires_in": 3600
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response):
            manager = OAuth2TokenManager()
            
            manager._tokens["test_provider"] = {
                "access_token": "old_token",
                "expires_at": time.time() - 100
            }
            
            token = manager.get_access_token(config)
            assert token == "new_token_456"

    def test_token_acquisition_failure(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider"
        )
        
        with patch('requests.post', side_effect=requests.RequestException("Network error")):
            manager = OAuth2TokenManager()
            
            with pytest.raises(RuntimeError, match="Failed to acquire OAuth2 token"):
                manager.get_access_token(config)

    def test_invalid_token_response(self):
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://example.com/token",
            provider_name="test_provider"
        )
        
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response):
            manager = OAuth2TokenManager()
            
            with pytest.raises(RuntimeError, match="Invalid token response"):
                manager.get_access_token(config)


class TestLLMOAuth2Integration:
    def test_llm_with_oauth2_config(self):
        config_data = {
            "oauth2_providers": {
                "custom": {
                    "client_id": "test_client",
                    "client_secret": "test_secret",
                    "token_url": "https://example.com/token"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            llm = LLM(
                model="custom/test-model",
                oauth2_config_path=config_path
            )
            
            assert "custom" in llm.oauth2_configs
            assert llm.oauth2_configs["custom"].client_id == "test_client"
        finally:
            Path(config_path).unlink()

    def test_llm_without_oauth2_config(self):
        llm = LLM(model="openai/gpt-3.5-turbo")
        assert llm.oauth2_configs == {}

    @patch('crewai.llm.litellm.completion')
    def test_llm_oauth2_token_injection(self, mock_completion):
        config_data = {
            "oauth2_providers": {
                "custom": {
                    "client_id": "test_client",
                    "client_secret": "test_secret",
                    "token_url": "https://example.com/token"
                }
            }
        }
        
        mock_completion.return_value = Mock(choices=[Mock(message=Mock(content="test response"))])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            with patch.object(OAuth2TokenManager, 'get_access_token', return_value="oauth_token_123"):
                llm = LLM(
                    model="custom/test-model",
                    oauth2_config_path=config_path
                )
                
                llm.call("test message")
                
                call_args = mock_completion.call_args
                assert call_args[1]['api_key'] == "oauth_token_123"
        finally:
            Path(config_path).unlink()

    @patch('crewai.llm.litellm.completion')
    def test_llm_oauth2_authentication_failure(self, mock_completion):
        config_data = {
            "oauth2_providers": {
                "custom": {
                    "client_id": "test_client",
                    "client_secret": "test_secret",
                    "token_url": "https://example.com/token"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            with patch.object(OAuth2TokenManager, 'get_access_token', side_effect=RuntimeError("Auth failed")):
                llm = LLM(
                    model="custom/test-model",
                    oauth2_config_path=config_path
                )
                
                with pytest.raises(RuntimeError, match="Auth failed"):
                    llm.call("test message")
        finally:
            Path(config_path).unlink()

    @patch('crewai.llm.litellm.completion')
    def test_llm_non_oauth2_provider_unchanged(self, mock_completion):
        mock_completion.return_value = Mock(choices=[Mock(message=Mock(content="test response"))])
        
        llm = LLM(
            model="openai/gpt-3.5-turbo",
            api_key="original_key"
        )
        
        llm.call("test message")
        
        call_args = mock_completion.call_args
        assert call_args[1]['api_key'] == "original_key"

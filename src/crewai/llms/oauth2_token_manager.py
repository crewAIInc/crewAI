import time
import logging
import requests
from threading import Lock
from typing import Dict, Any
from .oauth2_config import OAuth2Config
from .oauth2_errors import OAuth2AuthenticationError


class OAuth2TokenManager:
    def __init__(self):
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def get_access_token(self, config: OAuth2Config) -> str:
        """Get valid access token for the provider, refreshing if necessary"""
        with self._lock:
            return self._get_access_token_internal(config)
    
    def _get_access_token_internal(self, config: OAuth2Config) -> str:
        """Internal method to get access token (called within lock)"""
        provider_name = config.provider_name
        
        if provider_name in self._tokens:
            token_data = self._tokens[provider_name]
            if self._is_token_valid(token_data):
                logging.debug(f"Using cached OAuth2 token for provider {provider_name}")
                return token_data["access_token"]
        
        logging.info(f"Acquiring new OAuth2 token for provider {provider_name}")
        return self._acquire_new_token(config)
    
    def _is_token_valid(self, token_data: Dict[str, Any]) -> bool:
        """Check if token is still valid (not expired)"""
        if "expires_at" not in token_data:
            return False
        
        return time.time() < (token_data["expires_at"] - 60)
    
    def _acquire_new_token(self, config: OAuth2Config, retry_count: int = 3) -> str:
        """Acquire new access token using client credentials flow with retry logic"""
        for attempt in range(retry_count):
            try:
                return self._perform_token_request(config)
            except requests.RequestException as e:
                if attempt == retry_count - 1:
                    raise OAuth2AuthenticationError(
                        f"Failed to acquire OAuth2 token for {config.provider_name} after {retry_count} attempts: {e}",
                        original_error=e
                    )
                wait_time = 2 ** attempt
                logging.warning(f"OAuth2 token request failed for {config.provider_name}, retrying in {wait_time}s (attempt {attempt + 1}/{retry_count}): {e}")
                time.sleep(wait_time)
        
        raise OAuth2AuthenticationError(f"Unexpected error in token acquisition for {config.provider_name}")
    
    def _perform_token_request(self, config: OAuth2Config) -> str:
        """Perform the actual token request"""
        payload = {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
        
        if config.scope:
            payload["scope"] = config.scope
        
        try:
            response = requests.post(
                config.token_url,
                data=payload,
                timeout=30,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data["access_token"]
            
            expires_in = token_data.get("expires_in", 3600)
            self._tokens[config.provider_name] = {
                "access_token": access_token,
                "expires_at": time.time() + expires_in,
                "token_type": token_data.get("token_type", "Bearer")
            }
            
            logging.info(f"Successfully acquired OAuth2 token for {config.provider_name}, expires in {expires_in}s")
            return access_token
            
        except requests.RequestException:
            raise
        except KeyError as e:
            raise OAuth2AuthenticationError(f"Invalid token response from {config.provider_name}: missing {e}")

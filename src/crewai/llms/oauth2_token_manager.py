import time
import requests
from typing import Dict, Any
from .oauth2_config import OAuth2Config


class OAuth2TokenManager:
    def __init__(self):
        self._tokens: Dict[str, Dict[str, any]] = {}
    
    def get_access_token(self, config: OAuth2Config) -> str:
        """Get valid access token for the provider, refreshing if necessary"""
        provider_name = config.provider_name
        
        if provider_name in self._tokens:
            token_data = self._tokens[provider_name]
            if self._is_token_valid(token_data):
                return token_data["access_token"]
        
        return self._acquire_new_token(config)
    
    def _is_token_valid(self, token_data: Dict[str, Any]) -> bool:
        """Check if token is still valid (not expired)"""
        if "expires_at" not in token_data:
            return False
        
        return time.time() < (token_data["expires_at"] - 60)
    
    def _acquire_new_token(self, config: OAuth2Config) -> str:
        """Acquire new access token using client credentials flow"""
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
            
            return access_token
            
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to acquire OAuth2 token for {config.provider_name}: {e}")
        except KeyError as e:
            raise RuntimeError(f"Invalid token response from {config.provider_name}: missing {e}")

from pathlib import Path
from typing import Dict, List, Optional
import json
from pydantic import BaseModel, Field


class OAuth2Config(BaseModel):
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret") 
    token_url: str = Field(description="OAuth2 token endpoint URL")
    scope: Optional[str] = Field(default=None, description="OAuth2 scope")
    provider_name: str = Field(description="Custom provider name")
    refresh_token: Optional[str] = Field(default=None, description="OAuth2 refresh token")


class OAuth2ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("litellm_config.json")
    
    def load_config(self) -> Dict[str, OAuth2Config]:
        """Load OAuth2 configurations from litellm_config.json"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            oauth2_configs = {}
            for provider_name, config_data in data.get("oauth2_providers", {}).items():
                oauth2_configs[provider_name] = OAuth2Config(
                    provider_name=provider_name,
                    **config_data
                )
            
            return oauth2_configs
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid OAuth2 configuration in {self.config_path}: {e}")

from pathlib import Path
from typing import Dict, Optional
import json
import re
from pydantic import BaseModel, Field, field_validator
from .oauth2_errors import OAuth2ConfigurationError, OAuth2ValidationError


class OAuth2Config(BaseModel):
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret") 
    token_url: str = Field(description="OAuth2 token endpoint URL")
    scope: Optional[str] = Field(default=None, description="OAuth2 scope")
    provider_name: str = Field(description="Custom provider name")
    refresh_token: Optional[str] = Field(default=None, description="OAuth2 refresh token")

    @field_validator('token_url')
    @classmethod
    def validate_token_url(cls, v: str) -> str:
        """Validate that token_url is a valid HTTP/HTTPS URL."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(v):
            raise OAuth2ValidationError(f"Invalid token URL format: {v}")
        return v

    @field_validator('scope')
    @classmethod
    def validate_scope(cls, v: Optional[str]) -> Optional[str]:
        """Validate OAuth2 scope format."""
        if v:
            if '  ' in v:
                raise OAuth2ValidationError("Invalid scope format: scope cannot contain empty values")
        return v


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
            raise OAuth2ConfigurationError(f"Invalid OAuth2 configuration in {self.config_path}: {e}")

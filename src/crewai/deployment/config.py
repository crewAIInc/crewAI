import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator


class CrewConfig(BaseModel):
    """Configuration for a crew in a deployment."""
    name: str = Field(..., min_length=1)
    module_path: str = Field(..., min_length=1)
    class_name: str = Field(..., min_length=1)


class FlowConfig(BaseModel):
    """Configuration for a flow in a deployment."""
    name: str = Field(..., min_length=1)
    module_path: str = Field(..., min_length=1)
    class_name: str = Field(..., min_length=1)


class DeploymentConfig(BaseModel):
    """Main configuration for a CrewAI deployment."""
    name: str = Field(..., min_length=1)
    port: int = Field(..., gt=0, lt=65536)
    host: Optional[str] = Field(default="127.0.0.1")
    crews: List[CrewConfig] = Field(default_factory=list)
    flows: List[FlowConfig] = Field(default_factory=list)
    environment: List[str] = Field(default_factory=list)
    
    @validator('environment', pre=True)
    def parse_environment(cls, v):
        if not v:
            return []
        return v


class Config:
    """
    Configuration manager for CrewAI deployments.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        self.config = self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in configuration file: {e}")
                
    def _validate_config(self) -> DeploymentConfig:
        """Validate configuration using Pydantic."""
        try:
            return DeploymentConfig(**self._config_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
            
    @property
    def name(self) -> str:
        """Get deployment name."""
        return self.config.name
        
    @property
    def port(self) -> int:
        """Get server port."""
        return self.config.port
        
    @property
    def host(self) -> str:
        """Get host configuration."""
        return self.config.host
        
    @property
    def crews(self) -> List[CrewConfig]:
        """Get crews configuration."""
        return self.config.crews
        
    @property
    def flows(self) -> List[FlowConfig]:
        """Get flows configuration."""
        return self.config.flows
        
    @property
    def environment(self) -> List[str]:
        """Get environment variables configuration."""
        return self.config.environment

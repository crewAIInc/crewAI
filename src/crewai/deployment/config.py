from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

class Config:
    """
    Configuration for CrewAI deployments.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} not found")
            
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
            
    @property
    def name(self) -> str:
        """Get deployment name."""
        return self.config_data.get("name", "crewai-deployment")
        
    @property
    def port(self) -> int:
        """Get server port."""
        return int(self.config_data.get("port", 8000))
        
    @property
    def crews(self) -> List[Dict[str, Any]]:
        """Get crews configuration."""
        return self.config_data.get("crews", [])
        
    @property
    def flows(self) -> List[Dict[str, Any]]:
        """Get flows configuration."""
        return self.config_data.get("flows", [])

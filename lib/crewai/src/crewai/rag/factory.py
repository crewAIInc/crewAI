from typing import Any, Dict, Type, Union
import logging
from pydantic import BaseModel

from crewai.rag.core.base_client import BaseClient
from crewai.rag.config.types import RagConfigType
from crewai.rag.resilience.client import ResilientRAGClient

logger = logging.getLogger(__name__)

# Registry for RAG client factories
# Maps config types to their respective factory functions
_FACTORIES: Dict[Type[BaseModel], Any] = {}

def register_factory(config_type: Type[BaseModel], factory_fn: Any) -> None:
    """Register a new RAG client factory."""
    _FACTORIES[config_type] = factory_fn

def create_client(config: RagConfigType) -> BaseClient:
    """
    Create a RAG client based on the provided configuration.
    
    This function uses a registry of factories to instantiate the 
    appropriate client for the given configuration type.
    
    Args:
        config: The configuration object for the RAG client.
        
    Returns:
        A BaseClient instance.
        
    Raises:
        ValueError: If no factory is registered for the given config type.
    """
    factory_fn = _FACTORIES.get(type(config))
    if not factory_fn:
        raise ValueError(f"No RAG client factory registered for config type {type(config)}")
    
    raw_client = factory_fn(config)
    
    # Wrap the raw client with ResilientRAGClient to provide 
    # exponential backoff and graceful fallbacks.
    return ResilientRAGClient(raw_client)

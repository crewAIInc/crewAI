"""
Configuration management for A2A protocol in CrewAI.

This module provides configuration management for the A2A protocol implementation
in CrewAI, including default values and environment variable support.
"""

import os
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class A2AConfig(BaseModel):
    """Configuration for A2A protocol."""

    server_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the A2A server to.",
    )
    server_port: int = Field(
        default=8000,
        description="Port to bind the A2A server to.",
    )
    enable_cors: bool = Field(
        default=True,
        description="Whether to enable CORS for the A2A server.",
    )
    cors_origins: Optional[list[str]] = Field(
        default=None,
        description="CORS origins to allow. If None, all origins are allowed.",
    )
    
    client_timeout: int = Field(
        default=60,
        description="Timeout for A2A client requests in seconds.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for A2A authentication.",
    )
    
    task_ttl: int = Field(
        default=3600,
        description="Time-to-live for tasks in seconds.",
    )
    cleanup_interval: int = Field(
        default=300,
        description="Interval for cleaning up expired tasks in seconds.",
    )
    max_history_length: int = Field(
        default=100,
        description="Maximum number of messages to include in task history.",
    )

    @classmethod
    def from_env(cls) -> "A2AConfig":
        """Create a configuration from environment variables.
        
        Environment variables are prefixed with A2A_ and are uppercase.
        For example, A2A_SERVER_PORT=8080 will set server_port to 8080.
        
        Returns:
            A2AConfig: The configuration.
        """
        config_dict: Dict[str, Union[str, int, bool, list[str]]] = {}
        
        if "A2A_SERVER_HOST" in os.environ:
            config_dict["server_host"] = os.environ["A2A_SERVER_HOST"]
        if "A2A_SERVER_PORT" in os.environ:
            config_dict["server_port"] = int(os.environ["A2A_SERVER_PORT"])
        if "A2A_ENABLE_CORS" in os.environ:
            config_dict["enable_cors"] = os.environ["A2A_ENABLE_CORS"].lower() == "true"
        if "A2A_CORS_ORIGINS" in os.environ:
            config_dict["cors_origins"] = os.environ["A2A_CORS_ORIGINS"].split(",")
        
        if "A2A_CLIENT_TIMEOUT" in os.environ:
            config_dict["client_timeout"] = int(os.environ["A2A_CLIENT_TIMEOUT"])
        if "A2A_API_KEY" in os.environ:
            config_dict["api_key"] = os.environ["A2A_API_KEY"]
        
        if "A2A_TASK_TTL" in os.environ:
            config_dict["task_ttl"] = int(os.environ["A2A_TASK_TTL"])
        if "A2A_CLEANUP_INTERVAL" in os.environ:
            config_dict["cleanup_interval"] = int(os.environ["A2A_CLEANUP_INTERVAL"])
        if "A2A_MAX_HISTORY_LENGTH" in os.environ:
            config_dict["max_history_length"] = int(os.environ["A2A_MAX_HISTORY_LENGTH"])
        
        return cls(**config_dict)

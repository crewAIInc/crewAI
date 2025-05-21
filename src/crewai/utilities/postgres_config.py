"""
Utilities for PostgreSQL configuration in CrewAI.
"""

import os
from typing import Dict, Optional


def get_postgres_config() -> Dict[str, str]:
    """
    Returns a dictionary of PostgreSQL configuration from environment variables.
    
    The following environment variables are supported:
    - CREWAI_PG_HOST: PostgreSQL host (default: localhost)
    - CREWAI_PG_PORT: PostgreSQL port (default: 5432)
    - CREWAI_PG_USER: PostgreSQL username (default: postgres)
    - CREWAI_PG_PASSWORD: PostgreSQL password (default: empty string)
    - CREWAI_PG_DB: PostgreSQL database name (default: crewai)
    - CREWAI_PG_SCHEMA: PostgreSQL schema (default: public)
    - CREWAI_PG_TABLE: PostgreSQL table name (default: long_term_memories)
    - CREWAI_PG_MIN_POOL: Minimum connection pool size (default: 1)
    - CREWAI_PG_MAX_POOL: Maximum connection pool size (default: 5)
    - CREWAI_PG_ENABLE_POOL: Enable connection pooling (default: true)
    
    Returns:
        Dict[str, str]: Dictionary of PostgreSQL configuration
    """
    return {
        "host": os.environ.get("CREWAI_PG_HOST", "localhost"),
        "port": os.environ.get("CREWAI_PG_PORT", "5432"),
        "user": os.environ.get("CREWAI_PG_USER", "postgres"),
        "password": os.environ.get("CREWAI_PG_PASSWORD", ""),
        "db": os.environ.get("CREWAI_PG_DB", "crewai"),
        "schema": os.environ.get("CREWAI_PG_SCHEMA", "public"),
        "table": os.environ.get("CREWAI_PG_TABLE", "long_term_memories"),
        "min_pool": int(os.environ.get("CREWAI_PG_MIN_POOL", "1")),
        "max_pool": int(os.environ.get("CREWAI_PG_MAX_POOL", "5")),
        "enable_pool": os.environ.get("CREWAI_PG_ENABLE_POOL", "true").lower() == "true",
    }


def get_postgres_connection_string() -> Optional[str]:
    """
    Returns a PostgreSQL connection string from environment variables.
    
    The connection string is built from the individual components if not provided directly.
    If CREWAI_PG_CONNECTION_STRING is set, it will be used directly.
    
    Returns:
        Optional[str]: PostgreSQL connection string or None if not configured
    """
    # Check if a full connection string is provided
    direct_conn_string = os.environ.get("CREWAI_PG_CONNECTION_STRING")
    if direct_conn_string:
        return direct_conn_string
        
    # Otherwise, build from components
    config = get_postgres_config()
    
    # If no user/password provided, don't attempt to construct a connection string
    if not config["user"] and config["user"] == "postgres" and not config["db"]:
        return None
        
    # Build connection string
    if config["password"]:
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['db']}"
    else:
        return f"postgresql://{config['user']}@{config['host']}:{config['port']}/{config['db']}"
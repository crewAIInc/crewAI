"""Settings configuration using Pydantic."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === Heavy LLM (reasoning, planning, complex tasks) ===
    llm_provider: str = "lmstudio"
    llm_model: str = "qwen2.5-72b-instruct"
    llm_base_url: str = "http://192.168.178.181:1234/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # === Light LLM (simple tasks, tool calling) ===
    llm_light_provider: str = "lmstudio"
    llm_light_model: str = "qwen2.5-7b-instruct"
    llm_light_base_url: str = "http://192.168.178.181:1234/v1"
    llm_light_temperature: float = 0.3
    llm_light_max_tokens: int = 2048

    # === Embeddings (Ollama) ===
    embedder_provider: str = "ollama"
    embedder_model: str = "nomic-embed-text"
    embedder_base_url: str = "http://192.168.178.181:11434"

    # === Firecrawl (Web Scraping) ===
    firecrawl_api_url: str = "http://192.168.178.200:3002"
    firecrawl_api_key: str = ""  # Not needed for local instance

    # === Kraken Spot API ===
    kraken_api_key: str = ""
    kraken_api_secret: str = ""

    # === Kraken Futures API ===
    kraken_futures_api_key: str = ""
    kraken_futures_api_secret: str = ""

    # === Risk Settings ===
    kraken_balance_min_threshold: float = 0.000001
    max_leverage_spot: int = 1
    max_leverage_futures: int = 10
    risk_mode: Literal["normal", "reduced", "defensive"] = "normal"

    # === Crew Settings ===
    crew_verbose: bool = True
    crew_memory: bool = True
    crew_max_rpm: int = 60  # Rate limit per minute

    # === Chat LLM (for conversational crew interface) ===
    chat_llm_provider: str = "lmstudio"
    chat_llm_model: str = "fino1-8b-mlx"
    chat_llm_base_url: str = "http://192.168.178.181:1234/v1"
    chat_llm_temperature: float = 0.7


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

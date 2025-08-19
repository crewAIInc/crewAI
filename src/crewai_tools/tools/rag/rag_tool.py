import portalocker
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

from crewai.tools import BaseTool


def _fix_openai_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fix deprecated OpenAI configuration parameters in a config dictionary."""
    if not config:
        return config

    # Create a deep copy to avoid modifying the original
    fixed_config = deepcopy(config)

    def _is_azure_config(cfg: dict[str, Any]) -> bool:
        """Determine if this is an Azure OpenAI configuration."""
        # Check for explicit Azure indicators
        if cfg.get('provider') == 'azure_openai':
            return True

        # Check for Azure URLs in various fields
        for field in ['openai_api_base', 'base_url', 'api_base']:
            url = cfg.get(field, '')
            if isinstance(url, str) and 'azure' in url.lower():
                return True

        # Check if deployment is present (common in Azure configs)
        # But only if we also have other Azure indicators
        if 'deployment' in cfg:
            for field in ['openai_api_base', 'base_url', 'api_base']:
                url = cfg.get(field, '')
                if isinstance(url, str) and 'azure' in url.lower():
                    return True

        return False

    def _fix_config_recursively(cfg: dict[str, Any]) -> dict[str, Any]:
        """Recursively fix OpenAI config parameters."""
        if not isinstance(cfg, dict):
            return cfg

        # Only fix if this is definitely an Azure configuration
        if _is_azure_config(cfg):
            # Fix deprecated Azure OpenAI parameters
            if 'openai_api_base' in cfg and 'azure_endpoint' not in cfg:
                cfg['azure_endpoint'] = cfg.pop('openai_api_base')

            if 'base_url' in cfg and 'azure_endpoint' not in cfg:
                # Only convert base_url to azure_endpoint for Azure URLs
                base_url = cfg.get('base_url', '')
                if 'openai.azure.com' in base_url:
                    cfg['azure_endpoint'] = cfg.pop('base_url')

            # Handle deployment -> azure_deployment conversion for Azure configs
            if 'deployment' in cfg and 'azure_deployment' not in cfg:
                cfg['azure_deployment'] = cfg.pop('deployment')

        # For non-Azure configs, we might still need to handle some deprecated parameters
        # but we should NOT convert them to Azure format
        else:
            # For regular OpenAI configs, just remove the deprecated openai_api_base if present
            # since it's not valid for regular OpenAI (should use base_url instead)
            if 'openai_api_base' in cfg and 'base_url' not in cfg:
                # Only convert to base_url if it's NOT an Azure URL
                api_base = cfg.get('openai_api_base', '')
                if isinstance(api_base, str) and 'openai.azure.com' not in api_base:
                    cfg['base_url'] = cfg.pop('openai_api_base')

        # Recursively fix nested dictionaries
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = _fix_config_recursively(value)

        return cfg

    return _fix_config_recursively(fixed_config)


@contextmanager
def _temporarily_unset_env_vars():
    """Temporarily unset problematic environment variables that cause OpenAI validation issues."""
    problematic_vars = [
        'AZURE_API_BASE',
        'OPENAI_API_BASE',
        'OPENAI_BASE_URL'
    ]

    # Store original values
    original_values = {}
    for var in problematic_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]

    try:
        # Set the correct Azure environment variables if we had AZURE_API_BASE
        if 'AZURE_API_BASE' in original_values:
            os.environ['AZURE_OPENAI_ENDPOINT'] = original_values['AZURE_API_BASE']

        yield
    finally:
        # Restore original values
        for var, value in original_values.items():
            os.environ[var] = value

        # Clean up the temporary Azure endpoint we set
        if 'AZURE_API_BASE' in original_values and 'AZURE_OPENAI_ENDPOINT' in os.environ:
            if os.environ['AZURE_OPENAI_ENDPOINT'] == original_values['AZURE_API_BASE']:
                del os.environ['AZURE_OPENAI_ENDPOINT']


class Adapter(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge base with a question and return the answer."""

    @abstractmethod
    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add content to the knowledge base."""


class RagTool(BaseTool):
    class _AdapterPlaceholder(Adapter):
        def query(self, question: str) -> str:
            raise NotImplementedError

        def add(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError

    name: str = "Knowledge base"
    description: str = "A knowledge base that can be used to answer questions."
    summarize: bool = False
    adapter: Adapter = Field(default_factory=_AdapterPlaceholder)
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _set_default_adapter(self):
        if isinstance(self.adapter, RagTool._AdapterPlaceholder):
            from embedchain import App
            from crewai_tools.adapters.embedchain_adapter import EmbedchainAdapter

            with portalocker.Lock("crewai-rag-tool.lock", timeout=10):
                # Fix both environment variables and config parameters
                with _temporarily_unset_env_vars():
                    # Fix deprecated OpenAI parameters in config
                    fixed_config = _fix_openai_config(self.config)
                    app = App.from_config(config=fixed_config) if fixed_config else App()

            self.adapter = EmbedchainAdapter(
                embedchain_app=app, summarize=self.summarize
            )

        return self

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.adapter.add(*args, **kwargs)

    def _run(
        self,
        query: str,
    ) -> str:
        return f"Relevant Content:\n{self.adapter.query(query)}"

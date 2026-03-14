"""
MoltsPay Tool for CrewAI

Pay for AI services using USDC (gasless) via the x402 protocol.
Enables AI agents to autonomously purchase services from other agents.
"""

import os
from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MoltsPayToolSchema(BaseModel):
    """Input schema for MoltsPayTool."""

    provider_url: str = Field(
        ..., 
        description="Provider URL (e.g., 'https://example.com/api')"
    )
    service_id: str = Field(
        ..., 
        description="Service ID to call (e.g., 'text-to-video', 'image-generation')"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt or instructions for the service"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to image file for image-to-video or image processing services"
    )


class MoltsPayTool(BaseTool):
    """
    MoltsPayTool - Pay for AI services using crypto (USDC) via x402 protocol.

    This tool allows CrewAI agents to:
    - Discover AI services that accept MoltsPay
    - Pay for services using USDC (gasless, no ETH needed)
    - Execute paid AI services (video generation, image processing, etc.)

    The x402 protocol ensures pay-for-success: payment only completes if 
    the service delivers results.

    Dependencies:
        - moltspay (pip install moltspay)

    Environment Variables:
        - MOLTSPAY_WALLET_PATH: Path to wallet JSON file (optional, auto-creates)

    Example:
        ```python
        from crewai_tools import MoltsPayTool

        tool = MoltsPayTool()
        result = tool.run(
            provider_url="https://juai8.com/zen7",
            service_id="text-to-video",
            prompt="A cat dancing on a rainbow"
        )
        ```
    """

    name: str = "MoltsPay"
    description: str = (
        "Pay for AI services using USDC cryptocurrency (gasless). "
        "Use this to purchase video generation, image processing, transcription, "
        "and other AI services from providers that accept MoltsPay. "
        "Requires provider_url and service_id. Optional: prompt, image_path."
    )
    args_schema: Type[BaseModel] = MoltsPayToolSchema
    
    _client: Any = None

    def __init__(self, wallet_path: Optional[str] = None, **kwargs):
        """
        Initialize MoltsPayTool.

        Args:
            wallet_path: Path to MoltsPay wallet JSON file. 
                        If not provided, uses MOLTSPAY_WALLET_PATH env var
                        or default ~/.moltspay/wallet.json
        """
        super().__init__(**kwargs)
        self._wallet_path = wallet_path or os.environ.get(
            "MOLTSPAY_WALLET_PATH", 
            os.path.expanduser("~/.moltspay/wallet.json")
        )

    def _get_client(self):
        """Lazy-load MoltsPay client."""
        if self._client is None:
            try:
                from moltspay import MoltsPay
                self._client = MoltsPay(wallet_path=self._wallet_path)
            except ImportError:
                raise ImportError(
                    "MoltsPay is required for this tool. "
                    "Install with: pip install moltspay"
                )
        return self._client

    def _run(
        self,
        provider_url: str,
        service_id: str,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Execute a paid AI service via MoltsPay.

        Args:
            provider_url: The service provider's base URL
            service_id: The specific service to call
            prompt: Text prompt for the service
            image_path: Path to image file (for image-based services)

        Returns:
            Service response (URL to result, or result data)
        """
        try:
            client = self._get_client()
            
            # Build request params
            params = {}
            if prompt:
                params["prompt"] = prompt
            if image_path:
                params["image"] = image_path
            
            # Execute x402 payment + service call
            result = client.x402(
                url=f"{provider_url.rstrip('/')}/v1/{service_id}",
                method="POST",
                data=params
            )
            
            return str(result)
            
        except Exception as e:
            return f"MoltsPay error: {str(e)}"

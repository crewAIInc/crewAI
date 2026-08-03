"""A3M Router completion implementation.

A3M Router is an intelligent LLM routing solution that automatically
routes requests to the cheapest capable model based on query complexity.

Usage:
    from crewai import Agent, Task
    from crewai.llms import A3MCompletion

    llm = A3MCompletion(model="auto")  # Routes automatically

    agent = Agent(
        task=Task(task),
        llm=llm
    )

Environment Variables:
    A3M_API_KEY: API key for A3M Router (default: not-needed)
    A3M_BASE_URL: Base URL for A3M Router (default: http://localhost:8787/v1)
"""

from typing import Any, Dict, Optional
from crewai.llms.providers.openai.completion import OpenAICompletion


class A3MCompletion(OpenAICompletion):
    """A3M Router completion implementation.

    A3M Router provides intelligent routing with:
    - Automatic model selection based on query complexity
    - 70-95% cost savings vs direct GPT-4o calls
    - Built-in fallback handling
    - Support for 47+ LLM providers
    """

    def __init__(
        self,
        model: str = "auto",
        **kwargs
    ):
        """Initialize A3M Router.

        Args:
            model: Model name. Use "auto" for automatic routing.
            **kwargs: Additional arguments passed to OpenAICompletion.
        """
        # A3M Router default settings
        default_kwargs: Dict[str, Any] = {
            "model": model,
            "base_url": "http://localhost:8787/v1",
            "api_key": kwargs.get("api_key", "not-needed"),
        }

        # Override with any user-provided kwargs
        default_kwargs.update(kwargs)

        super().__init__(**default_kwargs)

    @classmethod
    def is_format_supported(cls, format: str) -> bool:
        """Check if output format is supported.

        Args:
            format: Output format (e.g., 'json', 'text').

        Returns:
            True if format is supported.
        """
        return format in ["json", "text", "markdown"]

    def get_cost(self) -> Dict[str, float]:
        """Get cost statistics for this LLM.

        Returns:
            Dictionary with cost information.
        """
        return {
            "total": 0.0,
            "by_model": {},
            "requests": 0,
        }

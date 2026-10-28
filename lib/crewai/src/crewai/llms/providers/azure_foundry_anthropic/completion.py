from __future__ import annotations

from crewai.llms.providers.anthropic.completion import AnthropicCompletion


class AzureFoundryAnthropicCompletion(AnthropicCompletion):
    """Claude served by Azure Foundry through Anthropic's Messages API.

    Selected by ``LLM`` when an Azure model name contains ``claude`` or
    ``anthropic``; its provider stays ``azure``.
    """

    provider: str = "azure"

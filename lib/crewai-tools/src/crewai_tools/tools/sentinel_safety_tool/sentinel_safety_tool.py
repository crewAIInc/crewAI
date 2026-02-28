"""
Sentinel Safety Tools for CrewAI

Provides AI safety guardrails for CrewAI agents using the THSP protocol
(Truth, Harm, Scope, Purpose).

Dependencies:
    - sentinelseed (pip install sentinelseed)
    - pydantic

See: https://sentinelseed.dev
"""

from typing import Any, Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SentinelSeedSchema(BaseModel):
    """Input schema for SentinelSafetyTool."""

    variant: Literal["minimal", "standard"] = Field(
        default="standard",
        description="Seed variant: 'minimal' (~450 tokens) or 'standard' (~1.4K tokens)",
    )


class SentinelAnalyzeSchema(BaseModel):
    """Input schema for SentinelAnalyzeTool."""

    content: str = Field(..., description="The content to analyze for safety")


class SentinelSafetyTool(BaseTool):
    """
    SentinelSafetyTool - Get alignment seeds for AI safety.

    Returns the Sentinel THSP (Truth, Harm, Scope, Purpose) alignment protocol
    that can be used as a system prompt to make LLMs safer.

    The THSP protocol evaluates requests through four gates:
    - Truth: Detects deception and manipulation
    - Harm: Identifies potential harmful content
    - Scope: Validates appropriate boundaries
    - Purpose: Requires legitimate benefit

    Dependencies:
        - sentinelseed (pip install sentinelseed)
    """

    name: str = "Sentinel Get Safety Seed"
    description: str = (
        "Get the Sentinel alignment seed - a system prompt that adds safety "
        "guardrails to any LLM using the THSP protocol (Truth, Harm, Scope, Purpose). "
        "Use 'minimal' for ~450 tokens or 'standard' for ~1.4K tokens. "
        "The seed can be used as a system prompt to improve AI safety."
    )
    args_schema: Type[BaseModel] = SentinelSeedSchema

    def _run(self, **kwargs: Any) -> str:
        """Get the Sentinel alignment seed."""
        variant = kwargs.get("variant", "standard")

        try:
            from sentinelseed import get_seed

            if variant not in ["minimal", "standard"]:
                return f"Error: Invalid variant '{variant}'. Use 'minimal' or 'standard'."

            seed = get_seed("v2", variant)
            return seed

        except ImportError:
            return (
                "Error: sentinelseed package not installed. "
                "Run: pip install sentinelseed"
            )
        except Exception as e:
            return f"Error getting seed: {str(e)}"


class SentinelAnalyzeTool(BaseTool):
    """
    SentinelAnalyzeTool - Analyze content for safety using THSP gates.

    Checks content against Sentinel's four-gate THSP protocol:
    - Truth: Detects deception patterns (fake, phishing, impersonation)
    - Harm: Identifies potential harmful content (violence, malware, weapons)
    - Scope: Flags bypass attempts (jailbreak, ignore instructions)
    - Purpose: Validates legitimate purpose

    Dependencies:
        - sentinelseed (pip install sentinelseed)
    """

    name: str = "Sentinel Analyze Content Safety"
    description: str = (
        "Analyze any text content for safety using Sentinel's THSP protocol. "
        "Returns whether the content is safe and which gates passed/failed. "
        "Use this to validate inputs, outputs, or any text before processing. "
        "Gates checked: Truth, Harm, Scope, Purpose."
    )
    args_schema: Type[BaseModel] = SentinelAnalyzeSchema

    def _run(self, **kwargs: Any) -> str:
        """Analyze content using THSP gates."""
        content = kwargs.get("content", "")

        if not content:
            return "Error: No content provided for analysis."

        try:
            from sentinelseed import SentinelGuard

            guard = SentinelGuard()
            analysis = guard.analyze(content)

            gates_str = ", ".join(
                f"{gate}: {status}" for gate, status in analysis.gates.items()
            )

            if analysis.safe:
                return (
                    f"SAFE - All gates passed.\n"
                    f"Gates: {gates_str}\n"
                    f"Confidence: {analysis.confidence:.0%}"
                )
            else:
                issues_str = ", ".join(analysis.issues) if analysis.issues else "Unknown"
                return (
                    f"UNSAFE - Safety check failed.\n"
                    f"Issues: {issues_str}\n"
                    f"Gates: {gates_str}\n"
                    f"Confidence: {analysis.confidence:.0%}"
                )

        except ImportError:
            return (
                "Error: sentinelseed package not installed. "
                "Run: pip install sentinelseed"
            )
        except Exception as e:
            return f"Error analyzing content: {str(e)}"

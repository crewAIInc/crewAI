"""
SwiftAPI-Enabled Tools for CrewAI

Wraps CrewAI tools with SwiftAPI attestation.
Every tool invocation requires cryptographic authorization before execution.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from crewai.tools.structured_tool import CrewStructuredTool

from .attestation import (
    AttestationError,
    AttestationProvider,
    AttestationResult,
    PolicyViolationError,
    SwiftAPIAttestationProvider,
)
from .config import SwiftAPIConfig

logger = logging.getLogger(__name__)


class SwiftAPIStructuredTool(CrewStructuredTool):
    """CrewAI structured tool with SwiftAPI attestation.

    Wraps a CrewStructuredTool to require attestation before every invocation.
    If attestation fails, the tool execution is blocked.

    Usage:
        # Wrap an existing tool
        original_tool = CrewStructuredTool.from_function(my_func)
        attested_tool = SwiftAPIStructuredTool.wrap(
            original_tool,
            swiftapi_key="swiftapi_live_..."
        )

        # Or create directly
        attested_tool = SwiftAPIStructuredTool(
            name="my_tool",
            description="Does something",
            args_schema=MyArgs,
            func=my_func,
            swiftapi_key="swiftapi_live_..."
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        func: Callable[..., Any],
        swiftapi_key: Optional[str] = None,
        config: Optional[SwiftAPIConfig] = None,
        attestation_provider: Optional[AttestationProvider] = None,
        agent_name: Optional[str] = None,
        crew_name: Optional[str] = None,
        result_as_answer: bool = False,
        max_usage_count: Optional[int] = None,
        current_usage_count: int = 0,
    ) -> None:
        """Initialize SwiftAPI-enabled tool.

        Args:
            name: Tool name
            description: Tool description
            args_schema: Pydantic model for arguments
            func: The function to execute
            swiftapi_key: SwiftAPI authority key (alternative to config)
            config: Full SwiftAPIConfig object
            attestation_provider: Custom attestation provider (for testing)
            agent_name: Name of the agent using this tool (for audit trail)
            crew_name: Name of the crew (for audit trail)
            result_as_answer: Whether to return output directly
            max_usage_count: Maximum tool usage limit
            current_usage_count: Current usage count
        """
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            func=func,
            result_as_answer=result_as_answer,
            max_usage_count=max_usage_count,
            current_usage_count=current_usage_count,
        )

        # Set up SwiftAPI config
        if config is not None:
            self._swiftapi_config = config
        else:
            self._swiftapi_config = SwiftAPIConfig(api_key=swiftapi_key)

        # Set up attestation provider
        if attestation_provider is not None:
            self._attestation_provider = attestation_provider
        elif self._swiftapi_config.is_configured:
            self._attestation_provider = SwiftAPIAttestationProvider(self._swiftapi_config)
        else:
            logger.warning(
                "SwiftAPI key not configured. All tool invocations will be blocked. "
                "Set SWIFTAPI_KEY or pass swiftapi_key parameter."
            )
            self._attestation_provider = None

        self._agent_name = agent_name
        self._crew_name = crew_name

    @classmethod
    def wrap(
        cls,
        tool: CrewStructuredTool,
        swiftapi_key: Optional[str] = None,
        config: Optional[SwiftAPIConfig] = None,
        attestation_provider: Optional[AttestationProvider] = None,
        agent_name: Optional[str] = None,
        crew_name: Optional[str] = None,
    ) -> SwiftAPIStructuredTool:
        """Wrap an existing CrewStructuredTool with SwiftAPI attestation.

        Args:
            tool: The tool to wrap
            swiftapi_key: SwiftAPI authority key
            config: Full SwiftAPIConfig
            attestation_provider: Custom provider (for testing)
            agent_name: Agent name for audit trail
            crew_name: Crew name for audit trail

        Returns:
            SwiftAPIStructuredTool wrapping the original
        """
        wrapped = cls(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=tool.func,
            swiftapi_key=swiftapi_key,
            config=config,
            attestation_provider=attestation_provider,
            agent_name=agent_name,
            crew_name=crew_name,
            result_as_answer=tool.result_as_answer,
            max_usage_count=tool.max_usage_count,
            current_usage_count=tool.current_usage_count,
        )
        wrapped._original_tool = tool._original_tool
        return wrapped

    def _format_intent(self, parsed_args: Dict[str, Any]) -> str:
        """Generate human-readable intent for the action."""
        intent_parts = [f"crewai tool '{self.name}'"]

        if self._agent_name:
            intent_parts.append(f"by agent '{self._agent_name}'")

        if self._crew_name:
            intent_parts.append(f"in crew '{self._crew_name}'")

        # Summarize args (truncate long values)
        if parsed_args:
            arg_summary = []
            for k, v in list(parsed_args.items())[:3]:
                v_str = str(v)[:50]
                if len(str(v)) > 50:
                    v_str += "..."
                arg_summary.append(f"{k}={v_str}")
            intent_parts.append(f"with {', '.join(arg_summary)}")

        return " ".join(intent_parts)

    def _build_context(self) -> Dict[str, Any]:
        """Build context for attestation request."""
        return {
            "tool_name": self.name,
            "agent_name": self._agent_name,
            "crew_name": self._crew_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _get_attestation(
        self,
        parsed_args: Dict[str, Any],
    ) -> AttestationResult:
        """Get attestation for tool invocation.

        Returns:
            AttestationResult if approved.

        Raises:
            PolicyViolationError: If action is denied.
            AttestationError: If attestation fails.
        """
        if self._attestation_provider is None:
            raise AttestationError(
                f"SwiftAPI not configured. Tool '{self.name}' blocked. "
                "Set SWIFTAPI_KEY environment variable or pass swiftapi_key parameter."
            )

        intent = self._format_intent(parsed_args)
        context = self._build_context()

        result = await self._attestation_provider.verify_action(
            action_type="tool_invocation",
            action_params={"tool": self.name, "args": parsed_args},
            intent=intent,
            context=context,
        )

        if self._swiftapi_config.verbose and result.approved:
            jti_short = result.jti[:12] if result.jti else "none"
            logger.info(f"\033[32m[SwiftAPI]\033[0m Approved: {self.name} (JTI: {jti_short}...)")

        return result

    def invoke(
        self,
        input: Union[str, dict],
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke tool with SwiftAPI attestation (synchronous).

        This wraps the parent invoke() with attestation verification.
        If attestation fails, a RuntimeError is raised.
        """
        parsed_args = self._parse_args(input)

        # Run attestation check
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to use a different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._get_attestation(parsed_args)
                    )
                    attestation = future.result()
            else:
                attestation = asyncio.run(self._get_attestation(parsed_args))
        except PolicyViolationError as e:
            error_msg = f"Tool '{self.name}' denied by policy: {e.denial_reason}"
            logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg) from e
        except AttestationError as e:
            error_msg = f"SwiftAPI attestation failed for '{self.name}': {e}"
            logger.error(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg) from e

        if not attestation.approved:
            error_msg = f"Tool '{self.name}' blocked by SwiftAPI: {attestation.reason}"
            logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg)

        # Attestation passed - call parent invoke
        # We already parsed args, so recreate the dict for parent
        return super().invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: Union[str, dict],
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronously invoke tool with SwiftAPI attestation.

        This wraps the parent ainvoke() with attestation verification.
        If attestation fails, a RuntimeError is raised.
        """
        parsed_args = self._parse_args(input)

        # Run attestation check
        try:
            attestation = await self._get_attestation(parsed_args)
        except PolicyViolationError as e:
            error_msg = f"Tool '{self.name}' denied by policy: {e.denial_reason}"
            logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg) from e
        except AttestationError as e:
            error_msg = f"SwiftAPI attestation failed for '{self.name}': {e}"
            logger.error(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg) from e

        if not attestation.approved:
            error_msg = f"Tool '{self.name}' blocked by SwiftAPI: {attestation.reason}"
            logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
            raise RuntimeError(error_msg)

        # Attestation passed - call parent ainvoke
        return await super().ainvoke(input, config, **kwargs)

    async def close(self) -> None:
        """Close the attestation provider."""
        if self._attestation_provider and hasattr(self._attestation_provider, "close"):
            await self._attestation_provider.close()


def wrap_tools(
    tools: List[CrewStructuredTool],
    swiftapi_key: Optional[str] = None,
    config: Optional[SwiftAPIConfig] = None,
    agent_name: Optional[str] = None,
    crew_name: Optional[str] = None,
) -> List[SwiftAPIStructuredTool]:
    """Wrap a list of tools with SwiftAPI attestation.

    Args:
        tools: List of CrewStructuredTool instances
        swiftapi_key: SwiftAPI authority key
        config: Full SwiftAPIConfig
        agent_name: Agent name for audit trail
        crew_name: Crew name for audit trail

    Returns:
        List of SwiftAPIStructuredTool instances
    """
    if config is None:
        config = SwiftAPIConfig(api_key=swiftapi_key)

    # Create a shared provider for efficiency
    provider = SwiftAPIAttestationProvider(config) if config.is_configured else None

    wrapped = []
    for tool in tools:
        wrapped_tool = SwiftAPIStructuredTool.wrap(
            tool,
            config=config,
            attestation_provider=provider,
            agent_name=agent_name,
            crew_name=crew_name,
        )
        wrapped.append(wrapped_tool)

    return wrapped

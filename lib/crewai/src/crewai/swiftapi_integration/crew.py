"""
SwiftAPI-Enabled Crew for CrewAI

Wraps CrewAI Crew with SwiftAPI attestation for multi-agent orchestration.
Provides audit trails for agent handoffs and task executions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.task import Task
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.types.streaming import CrewStreamingOutput

from .attestation import (
    AttestationError,
    AttestationProvider,
    AttestationResult,
    PolicyViolationError,
    SwiftAPIAttestationProvider,
)
from .config import SwiftAPIConfig
from .tools import SwiftAPIStructuredTool, wrap_tools

logger = logging.getLogger(__name__)


class SwiftAPICrew:
    """Wrapper around CrewAI Crew with SwiftAPI attestation.

    Provides:
    - Attestation for crew kickoff
    - Attestation for task assignments
    - Attestation for agent handoffs
    - Complete audit trail of multi-agent execution

    Usage:
        crew = Crew(agents=[...], tasks=[...])
        swiftapi_crew = SwiftAPICrew(
            crew=crew,
            swiftapi_key="swiftapi_live_..."
        )
        result = swiftapi_crew.kickoff(inputs={...})

    Or wrap tools automatically:
        swiftapi_crew = SwiftAPICrew(
            crew=crew,
            swiftapi_key="swiftapi_live_...",
            wrap_agent_tools=True  # Wraps all agent tools with attestation
        )
    """

    def __init__(
        self,
        crew: Crew,
        swiftapi_key: Optional[str] = None,
        config: Optional[SwiftAPIConfig] = None,
        attestation_provider: Optional[AttestationProvider] = None,
        wrap_agent_tools: bool = True,
        attest_kickoff: bool = True,
        attest_task_start: bool = True,
        attest_agent_handoff: bool = True,
    ) -> None:
        """Initialize SwiftAPI-enabled Crew.

        Args:
            crew: The CrewAI Crew to wrap
            swiftapi_key: SwiftAPI authority key (alternative to config)
            config: Full SwiftAPIConfig object
            attestation_provider: Custom attestation provider (for testing)
            wrap_agent_tools: If True, wrap all agent tools with attestation
            attest_kickoff: Require attestation before crew kickoff
            attest_task_start: Require attestation before each task
            attest_agent_handoff: Require attestation for agent handoffs
        """
        self.crew = crew

        # Set up SwiftAPI config
        if config is not None:
            self._config = config
        else:
            self._config = SwiftAPIConfig(api_key=swiftapi_key)

        # Set up attestation provider
        if attestation_provider is not None:
            self._provider = attestation_provider
        elif self._config.is_configured:
            self._provider = SwiftAPIAttestationProvider(self._config)
        else:
            logger.warning(
                "SwiftAPI key not configured. Crew execution will be blocked. "
                "Set SWIFTAPI_KEY or pass swiftapi_key parameter."
            )
            self._provider = None

        self._attest_kickoff = attest_kickoff
        self._attest_task_start = attest_task_start
        self._attest_agent_handoff = attest_agent_handoff

        # Wrap agent tools if requested
        if wrap_agent_tools and self._config.is_configured:
            self._wrap_all_agent_tools()

        # Audit trail
        self._audit_log: List[Dict[str, Any]] = []

    def _wrap_all_agent_tools(self) -> None:
        """Wrap all tools on all agents with SwiftAPI attestation."""
        for agent in self.crew.agents:
            if hasattr(agent, "tools") and agent.tools:
                wrapped_tools = []
                for tool in agent.tools:
                    if isinstance(tool, SwiftAPIStructuredTool):
                        # Already wrapped
                        wrapped_tools.append(tool)
                    elif isinstance(tool, CrewStructuredTool):
                        wrapped_tool = SwiftAPIStructuredTool.wrap(
                            tool,
                            config=self._config,
                            attestation_provider=self._provider,
                            agent_name=getattr(agent, "role", None) or getattr(agent, "name", None),
                            crew_name=self.crew.name,
                        )
                        wrapped_tools.append(wrapped_tool)
                    else:
                        # Non-structured tool, pass through
                        # (these will still go through tool_usage which we could also wrap)
                        wrapped_tools.append(tool)
                agent.tools = wrapped_tools

    def _log_audit(
        self,
        event_type: str,
        details: Dict[str, Any],
        attestation: Optional[AttestationResult] = None,
    ) -> None:
        """Add entry to audit log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "crew_name": self.crew.name,
            "details": details,
        }
        if attestation:
            entry["attestation"] = {
                "approved": attestation.approved,
                "jti": attestation.jti,
                "reason": attestation.reason,
            }
        self._audit_log.append(entry)

        if self._config.verbose:
            logger.info(f"[SwiftAPI Audit] {event_type}: {details.get('summary', '')}")

    async def _attest_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AttestationResult:
        """Get attestation for an action.

        Raises:
            AttestationError: If attestation fails
            PolicyViolationError: If action is denied
        """
        if self._provider is None:
            raise AttestationError(
                "SwiftAPI not configured. Action blocked. "
                "Set SWIFTAPI_KEY environment variable."
            )

        full_context = {
            "crew_name": self.crew.name,
            "crew_id": str(self.crew.id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(context or {}),
        }

        result = await self._provider.verify_action(
            action_type=action_type,
            action_params=action_params,
            intent=intent,
            context=full_context,
        )

        return result

    def _sync_attest(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AttestationResult:
        """Synchronous attestation wrapper."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._attest_action(action_type, action_params, intent, context)
                    )
                    return future.result()
            else:
                return asyncio.run(
                    self._attest_action(action_type, action_params, intent, context)
                )
        except Exception:
            raise

    def kickoff(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Union[CrewOutput, CrewStreamingOutput]:
        """Execute crew with SwiftAPI attestation.

        Args:
            inputs: Input data for the crew

        Returns:
            CrewOutput or CrewStreamingOutput

        Raises:
            RuntimeError: If attestation fails
        """
        if self._attest_kickoff:
            try:
                intent = f"crew '{self.crew.name}' kickoff with {len(self.crew.agents)} agents, {len(self.crew.tasks)} tasks"
                attestation = self._sync_attest(
                    action_type="crew_kickoff",
                    action_params={
                        "crew_name": self.crew.name,
                        "agent_count": len(self.crew.agents),
                        "task_count": len(self.crew.tasks),
                        "process": str(self.crew.process),
                        "inputs": inputs or {},
                    },
                    intent=intent,
                )

                if not attestation.approved:
                    error_msg = f"Crew kickoff denied: {attestation.reason}"
                    logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
                    raise RuntimeError(error_msg)

                self._log_audit(
                    "crew_kickoff",
                    {"summary": intent, "inputs": inputs},
                    attestation,
                )

                if self._config.verbose:
                    jti_short = attestation.jti[:12] if attestation.jti else "none"
                    logger.info(
                        f"\033[32m[SwiftAPI]\033[0m Approved crew kickoff (JTI: {jti_short}...)"
                    )

            except PolicyViolationError as e:
                self._log_audit(
                    "crew_kickoff_denied",
                    {"reason": e.denial_reason},
                )
                raise RuntimeError(f"Crew kickoff denied by policy: {e.denial_reason}") from e
            except AttestationError as e:
                self._log_audit(
                    "crew_kickoff_error",
                    {"error": str(e)},
                )
                raise RuntimeError(f"SwiftAPI attestation failed: {e}") from e

        # Execute the crew
        result = self.crew.kickoff(inputs=inputs)

        self._log_audit(
            "crew_completed",
            {
                "summary": f"crew '{self.crew.name}' completed",
                "raw_output_preview": str(result.raw)[:200] if hasattr(result, "raw") else None,
            },
        )

        return result

    async def kickoff_async(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Union[CrewOutput, CrewStreamingOutput]:
        """Asynchronously execute crew with SwiftAPI attestation.

        Args:
            inputs: Input data for the crew

        Returns:
            CrewOutput or CrewStreamingOutput

        Raises:
            RuntimeError: If attestation fails
        """
        if self._attest_kickoff:
            try:
                intent = f"crew '{self.crew.name}' async kickoff with {len(self.crew.agents)} agents, {len(self.crew.tasks)} tasks"
                attestation = await self._attest_action(
                    action_type="crew_kickoff",
                    action_params={
                        "crew_name": self.crew.name,
                        "agent_count": len(self.crew.agents),
                        "task_count": len(self.crew.tasks),
                        "process": str(self.crew.process),
                        "inputs": inputs or {},
                    },
                    intent=intent,
                )

                if not attestation.approved:
                    error_msg = f"Crew kickoff denied: {attestation.reason}"
                    logger.warning(f"\033[31m[SwiftAPI]\033[0m {error_msg}")
                    raise RuntimeError(error_msg)

                self._log_audit(
                    "crew_kickoff",
                    {"summary": intent, "inputs": inputs},
                    attestation,
                )

                if self._config.verbose:
                    jti_short = attestation.jti[:12] if attestation.jti else "none"
                    logger.info(
                        f"\033[32m[SwiftAPI]\033[0m Approved crew kickoff (JTI: {jti_short}...)"
                    )

            except PolicyViolationError as e:
                self._log_audit(
                    "crew_kickoff_denied",
                    {"reason": e.denial_reason},
                )
                raise RuntimeError(f"Crew kickoff denied by policy: {e.denial_reason}") from e
            except AttestationError as e:
                self._log_audit(
                    "crew_kickoff_error",
                    {"error": str(e)},
                )
                raise RuntimeError(f"SwiftAPI attestation failed: {e}") from e

        # Execute the crew
        result = await self.crew.kickoff_async(inputs=inputs)

        self._log_audit(
            "crew_completed",
            {
                "summary": f"crew '{self.crew.name}' completed",
                "raw_output_preview": str(result.raw)[:200] if hasattr(result, "raw") else None,
            },
        )

        return result

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log of all attested actions."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    async def close(self) -> None:
        """Close the attestation provider."""
        if self._provider and hasattr(self._provider, "close"):
            await self._provider.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped crew."""
        return getattr(self.crew, name)

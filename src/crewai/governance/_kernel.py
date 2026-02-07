# Copyright (c) Agent-OS Contributors. All rights reserved.
# Licensed under the MIT License.
"""Kernel-level governance for CrewAI agents and crews."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from crewai import Agent, Crew, Task
except ImportError:
    Agent = Any
    Crew = Any
    Task = Any


class ViolationType(Enum):
    """Types of policy violations."""

    TOOL_BLOCKED = "tool_blocked"
    TOOL_LIMIT_EXCEEDED = "tool_limit_exceeded"
    ITERATION_LIMIT_EXCEEDED = "iteration_limit_exceeded"
    MESSAGE_BLOCKED = "message_blocked"
    MESSAGE_LIMIT_EXCEEDED = "message_limit_exceeded"
    TIMEOUT = "timeout"
    CONTENT_FILTERED = "content_filtered"


@dataclass
class PolicyViolation:
    """Represents a policy violation event."""

    violation_type: ViolationType
    policy_name: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    task_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit trail event."""

    event_type: str
    timestamp: datetime
    agent_name: Optional[str]
    task_name: Optional[str]
    details: Dict[str, Any]


@dataclass
class GovernancePolicy:
    """Policy configuration for crew governance.

    Attributes:
        max_tool_calls: Maximum tool invocations per task.
        max_iterations: Maximum agent iterations per task.
        max_execution_time: Maximum seconds per task.
        blocked_patterns: Regex patterns to block in outputs.
        blocked_tools: Tools that cannot be used.
        allowed_tools: If set, only these tools can be used.
        log_all_actions: Log all agent actions.
        max_output_length: Maximum output length in characters.
        
    Note:
        max_tool_calls and max_iterations are tracked but enforcement
        requires CrewAI callback integration (future enhancement).
    """

    max_tool_calls: int = 50
    max_iterations: int = 25
    max_execution_time: int = 600
    blocked_patterns: List[str] = field(default_factory=list)
    blocked_tools: List[str] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    log_all_actions: bool = True
    max_output_length: int = 100_000

    def __post_init__(self):
        """Compile regex patterns."""
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]


class GovernedAgent:
    """Wraps a CrewAI Agent with governance policies.

    Example:
        ```python
        from crewai import Agent
        from crewai.governance import GovernedAgent, GovernancePolicy

        policy = GovernancePolicy(
            max_tool_calls=10,
            blocked_tools=["shell_tool"],
        )

        researcher = Agent(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
        )

        governed_researcher = GovernedAgent(researcher, policy)
        ```
    """

    def __init__(
        self,
        agent: Agent,
        policy: GovernancePolicy,
        on_violation: Optional[Callable[[PolicyViolation], None]] = None,
    ):
        """Initialize governed agent.

        Args:
            agent: The CrewAI agent to govern.
            policy: Governance policy to enforce.
            on_violation: Callback when violations occur.
        """
        self.agent = agent
        self.policy = policy
        self.on_violation = on_violation
        self._tool_calls = 0
        self._iterations = 0
        self._start_time: Optional[float] = None
        self._violations: List[PolicyViolation] = []
        self._audit_log: List[AuditEvent] = []

        # Wrap agent's execute method
        self._wrap_execution()

    def _wrap_execution(self):
        """Wrap the agent's execution to add governance."""
        import logging
        logger = logging.getLogger(__name__)
        
        original_execute = getattr(self.agent, "execute_task", None)
        if original_execute is None:
            agent_name = getattr(self.agent, "role", "unknown")
            logger.warning(
                "GovernedAgent: Agent '%s' lacks 'execute_task' method. "
                "Governance will NOT be applied to this agent.",
                agent_name,
            )
            return

        @wraps(original_execute)
        def governed_execute(task: Any, context: Optional[str] = None, tools: Optional[List] = None):
            self._start_time = time.time()
            self._tool_calls = 0
            self._iterations = 0

            # Filter tools based on policy
            if tools:
                tools = self._filter_tools(tools)

            # Check tool call limit before execution
            if self.policy.max_tool_calls > 0 and self._tool_calls >= self.policy.max_tool_calls:
                self._record_violation(
                    ViolationType.TOOL_LIMIT_EXCEEDED,
                    f"Tool calls ({self._tool_calls}) would exceed limit ({self.policy.max_tool_calls})",
                )

            # Execute with governance
            try:
                result = original_execute(task, context, tools)
                
                # Increment iteration counter
                self._iterations += 1
                
                # Check iteration limit
                if self.policy.max_iterations > 0 and self._iterations > self.policy.max_iterations:
                    self._record_violation(
                        ViolationType.ITERATION_LIMIT_EXCEEDED,
                        f"Iterations ({self._iterations}) exceeded limit ({self.policy.max_iterations})",
                    )
                
                # Check output for violations
                result = self._check_output(result, task)
                return result
            finally:
                self._log_event("task_completed", task=task)

        self.agent.execute_task = governed_execute

    def _filter_tools(self, tools: List[Any]) -> List[Any]:
        """Filter tools based on policy."""
        filtered = []
        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))

            # Check blocked list
            if tool_name in self.policy.blocked_tools:
                self._record_violation(
                    ViolationType.TOOL_BLOCKED,
                    f"Tool '{tool_name}' is blocked by policy",
                    tool_name=tool_name,
                )
                continue

            # Check allowed list
            if (
                self.policy.allowed_tools is not None
                and tool_name not in self.policy.allowed_tools
            ):
                self._record_violation(
                    ViolationType.TOOL_BLOCKED,
                    f"Tool '{tool_name}' not in allowed list",
                    tool_name=tool_name,
                )
                continue

            filtered.append(tool)
            
            # Track tool call
            self._tool_calls += 1
            
            # Check tool call limit
            if self.policy.max_tool_calls > 0 and self._tool_calls > self.policy.max_tool_calls:
                self._record_violation(
                    ViolationType.TOOL_LIMIT_EXCEEDED,
                    f"Tool calls ({self._tool_calls}) exceeded limit ({self.policy.max_tool_calls})",
                )

        return filtered

    def _check_output(self, output: Any, task: Any) -> Any:
        """Check output for policy violations.
        
        Note: For non-string outputs, violations are detected and logged,
        but the original object is returned to preserve type compatibility.
        Violations can be retrieved via get_violations(). For string outputs,
        blocked content is replaced with [BLOCKED] and length is truncated.
        """
        if output is None:
            return output

        output_str = str(output)
        was_modified = False

        # Check length
        if len(output_str) > self.policy.max_output_length:
            self._record_violation(
                ViolationType.CONTENT_FILTERED,
                f"Output exceeds max length ({len(output_str)} > {self.policy.max_output_length})",
            )
            output_str = output_str[: self.policy.max_output_length]
            was_modified = True

        # Check patterns
        for pattern in self.policy._compiled_patterns:
            if pattern.search(output_str):
                self._record_violation(
                    ViolationType.CONTENT_FILTERED,
                    f"Output contains blocked pattern: {pattern.pattern}",
                    pattern=pattern.pattern,
                )
                output_str = pattern.sub("[BLOCKED]", output_str)
                was_modified = True

        # Return sanitized string, or log warning for non-string outputs
        if isinstance(output, str):
            return output_str
        elif was_modified:
            # Non-string output with violations - log warning but return original object
            # to avoid breaking downstream consumers that expect specific types.
            # The violation has already been recorded; callers can check get_violations()
            # if they need to know about content issues.
            logger.warning(
                "Content violation detected in non-string output (type: %s). "
                "Returning original object to preserve type. Violations: %d",
                type(output).__name__,
                len(self._violations)
            )
            return output
        return output

    def _record_violation(
        self,
        violation_type: ViolationType,
        description: str,
        **details: Any,
    ):
        """Record a policy violation."""
        violation = PolicyViolation(
            violation_type=violation_type,
            policy_name=violation_type.value,
            description=description,
            agent_name=getattr(self.agent, "role", "unknown"),
            details=details,
        )
        self._violations.append(violation)

        if self.on_violation:
            self.on_violation(violation)

        self._log_event(
            "violation",
            violation_type=violation_type.value,
            description=description,
        )

    def _log_event(self, event_type: str, **details: Any):
        """Log an audit event."""
        if not self.policy.log_all_actions:
            return

        # Extract task_name from either 'task_name' key or 'task' object
        task_name = details.get("task_name")
        if task_name is None and "task" in details:
            task_obj = details["task"]
            task_name = getattr(task_obj, "description", None) or getattr(task_obj, "name", None)

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            agent_name=getattr(self.agent, "role", "unknown"),
            task_name=task_name,
            details=details,
        )
        self._audit_log.append(event)

    @property
    def violations(self) -> List[PolicyViolation]:
        """Get all violations."""
        return self._violations.copy()

    @property
    def audit_log(self) -> List[AuditEvent]:
        """Get audit log."""
        return self._audit_log.copy()


class GovernedCrew:
    """Wraps a CrewAI Crew with governance policies.

    Example:
        ```python
        from crewai import Agent, Crew, Task
        from crewai.governance import GovernedCrew, GovernancePolicy

        policy = GovernancePolicy(
            max_tool_calls=20,
            max_iterations=15,
            blocked_patterns=["DROP TABLE", "rm -rf"],
        )

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
        )

        governed_crew = GovernedCrew(crew, policy)
        result = governed_crew.kickoff()
        print(f"Violations: {len(governed_crew.violations)}")
        ```
    """

    def __init__(
        self,
        crew: Crew,
        policy: GovernancePolicy,
        on_violation: Optional[Callable[[PolicyViolation], None]] = None,
    ):
        """Initialize governed crew.

        Args:
            crew: The CrewAI crew to govern.
            policy: Governance policy to enforce.
            on_violation: Callback when violations occur.
        """
        self.crew = crew
        self.policy = policy
        self.on_violation = on_violation
        self._violations: List[PolicyViolation] = []
        self._audit_log: List[AuditEvent] = []
        self._governed_agents: List[GovernedAgent] = []

        # Wrap all agents
        self._wrap_agents()

    def _wrap_agents(self):
        """Wrap all crew agents with governance."""
        for i, agent in enumerate(self.crew.agents):
            governed = GovernedAgent(agent, self.policy, self._handle_violation)
            self._governed_agents.append(governed)
            # Note: agents are already wrapped in place

    def _handle_violation(self, violation: PolicyViolation):
        """Handle violations from governed agents."""
        self._violations.append(violation)
        if self.on_violation:
            self.on_violation(violation)

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the crew with governance.

        Args:
            inputs: Optional inputs for the crew.

        Returns:
            Crew execution result.
            
        Note:
            The max_execution_time check is performed after execution completes
            and records a violation for audit purposes. For real-time timeout
            enforcement, implement async execution with asyncio.timeout or
            similar mechanisms at the application level.
        """
        self._log_event("crew_started", inputs=inputs)
        start_time = time.time()

        try:
            result = self.crew.kickoff(inputs=inputs)

            # Check execution time (audit-only - records violation post-execution)
            elapsed = time.time() - start_time
            if elapsed > self.policy.max_execution_time:
                self._record_violation(
                    ViolationType.TIMEOUT,
                    f"Execution time ({elapsed:.1f}s) exceeded limit ({self.policy.max_execution_time}s)",
                )

            return result
        finally:
            self._log_event(
                "crew_completed",
                duration=time.time() - start_time,
                violations=len(self._violations),
            )

    def _record_violation(
        self,
        violation_type: ViolationType,
        description: str,
        **details: Any,
    ):
        """Record a crew-level violation."""
        violation = PolicyViolation(
            violation_type=violation_type,
            policy_name=violation_type.value,
            description=description,
            details=details,
        )
        self._violations.append(violation)

        if self.on_violation:
            self.on_violation(violation)

        # Log crew-level violations to audit trail
        self._log_event(
            "violation",
            violation_type=violation_type.value,
            description=description,
            **details,
        )

    def _log_event(self, event_type: str, **details: Any):
        """Log audit event."""
        if not self.policy.log_all_actions:
            return

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            agent_name=None,
            task_name=None,
            details=details,
        )
        self._audit_log.append(event)

    @property
    def violations(self) -> List[PolicyViolation]:
        """Get all violations from crew and agents.
        
        Note: Agent violations are already collected via the callback,
        so we only return crew-level violations here. Agent violations
        are forwarded to self._violations via _handle_violation.
        """
        return self._violations.copy()

    @property
    def audit_log(self) -> List[AuditEvent]:
        """Get full audit log."""
        all_events = self._audit_log.copy()
        for agent in self._governed_agents:
            all_events.extend(agent.audit_log)
        return sorted(all_events, key=lambda e: e.timestamp)

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of governance audit."""
        return {
            "total_violations": len(self.violations),
            "violations_by_type": self._count_by_type(),
            "total_events": len(self.audit_log),
            "agents": [getattr(a.agent, "role", "unknown") for a in self._governed_agents],
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count violations by type."""
        counts: Dict[str, int] = {}
        for v in self.violations:
            key = v.violation_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

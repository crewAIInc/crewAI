"""
Accountability logging and tracking system.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.models import AccountabilityRecord
from crewai.task import Task


class AccountabilityLogger:
    """Logs and tracks agent actions for accountability."""

    def __init__(self):
        self.records: list[AccountabilityRecord] = []
        self.agent_records: dict[str, list[AccountabilityRecord]] = defaultdict(list)
        self._setup_event_listeners()

    def log_action(
        self,
        agent: BaseAgent,
        action_type: str,
        action_description: str,
        task: Task | None = None,
        context: dict[str, Any] | None = None
    ) -> AccountabilityRecord:
        """Log an agent action."""
        agent_id = self._get_agent_id(agent)
        task_id = str(task.id) if task else None

        record = AccountabilityRecord(
            agent_id=agent_id,
            action_type=action_type,
            action_description=action_description,
            task_id=task_id,
            context=context or {}
        )

        self.records.append(record)
        self.agent_records[agent_id].append(record)

        return record

    def log_decision(
        self,
        agent: BaseAgent,
        decision: str,
        reasoning: str,
        task: Task | None = None,
        alternatives_considered: list[str] | None = None
    ) -> AccountabilityRecord:
        """Log an agent decision with reasoning."""
        context = {
            "reasoning": reasoning,
            "alternatives_considered": alternatives_considered or []
        }

        return self.log_action(
            agent=agent,
            action_type="decision",
            action_description=decision,
            task=task,
            context=context
        )

    def log_delegation(
        self,
        delegating_agent: BaseAgent,
        receiving_agent: BaseAgent,
        task: Task,
        delegation_reason: str
    ) -> AccountabilityRecord:
        """Log task delegation between agents."""
        context = {
            "receiving_agent_id": self._get_agent_id(receiving_agent),
            "receiving_agent_role": receiving_agent.role,
            "delegation_reason": delegation_reason
        }

        return self.log_action(
            agent=delegating_agent,
            action_type="delegation",
            action_description=f"Delegated task to {receiving_agent.role}",
            task=task,
            context=context
        )

    def log_task_completion(
        self,
        agent: BaseAgent,
        task: Task,
        success: bool,
        outcome_description: str,
        completion_time: float | None = None
    ) -> AccountabilityRecord:
        """Log task completion with outcome."""
        context = {
            "completion_time": completion_time,
            "task_description": task.description
        }

        record = self.log_action(
            agent=agent,
            action_type="task_completion",
            action_description=f"Completed task: {task.description[:100]}...",
            task=task,
            context=context
        )

        record.set_outcome(outcome_description, success)
        return record

    def get_agent_records(
        self,
        agent: BaseAgent,
        action_type: str | None = None,
        since: datetime | None = None
    ) -> list[AccountabilityRecord]:
        """Get accountability records for a specific agent."""
        agent_id = self._get_agent_id(agent)
        records = self.agent_records.get(agent_id, [])

        if action_type:
            records = [r for r in records if r.action_type == action_type]

        if since:
            records = [r for r in records if r.timestamp >= since]

        return records

    def get_task_records(self, task: Task) -> list[AccountabilityRecord]:
        """Get all accountability records related to a specific task."""
        task_id = str(task.id)
        return [r for r in self.records if r.task_id == task_id]

    def get_delegation_chain(self, task: Task) -> list[AccountabilityRecord]:
        """Get the delegation chain for a task."""
        task_records = self.get_task_records(task)
        delegation_records = [r for r in task_records if r.action_type == "delegation"]

        delegation_records.sort(key=lambda r: r.timestamp)
        return delegation_records

    def generate_accountability_report(
        self,
        agent: BaseAgent | None = None,
        time_period: timedelta | None = None
    ) -> dict[str, Any]:
        """Generate an accountability report."""
        since = datetime.utcnow() - time_period if time_period else None

        if agent:
            records = self.get_agent_records(agent, since=since)
            agent_id = self._get_agent_id(agent)
        else:
            records = self.records
            if since:
                records = [r for r in records if r.timestamp >= since]
            agent_id = "all_agents"

        action_counts = defaultdict(int)
        success_counts = defaultdict(int)
        failure_counts = defaultdict(int)

        for record in records:
            action_counts[record.action_type] += 1
            if record.success is True:
                success_counts[record.action_type] += 1
            elif record.success is False:
                failure_counts[record.action_type] += 1

        success_rates = {}
        for action_type in action_counts:
            total = success_counts[action_type] + failure_counts[action_type]
            if total > 0:
                success_rates[action_type] = success_counts[action_type] / total
            else:
                success_rates[action_type] = None

        return {
            "agent_id": agent_id,
            "report_period": {
                "start": since.isoformat() if since else None,
                "end": datetime.utcnow().isoformat()
            },
            "total_records": len(records),
            "action_counts": dict(action_counts),
            "success_counts": dict(success_counts),
            "failure_counts": dict(failure_counts),
            "success_rates": success_rates,
            "recent_actions": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "action_type": r.action_type,
                    "description": r.action_description,
                    "success": r.success
                }
                for r in sorted(records, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }

    def _setup_event_listeners(self) -> None:
        """Set up event listeners for automatic logging."""

    def _get_agent_id(self, agent: BaseAgent) -> str:
        """Get a unique identifier for an agent."""
        return f"{agent.role}_{id(agent)}"

"""
Main responsibility system that coordinates all components.
"""

from datetime import datetime, timedelta
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.accountability import AccountabilityLogger
from crewai.responsibility.assignment import (
    AssignmentStrategy,
    ResponsibilityCalculator,
)
from crewai.responsibility.hierarchy import CapabilityHierarchy
from crewai.responsibility.models import (
    AgentCapability,
    ResponsibilityAssignment,
    TaskRequirement,
)
from crewai.responsibility.performance import PerformanceTracker
from crewai.task import Task


class ResponsibilitySystem:
    """Main system that coordinates all responsibility tracking components."""

    def __init__(self):
        self.hierarchy = CapabilityHierarchy()
        self.calculator = ResponsibilityCalculator(self.hierarchy)
        self.accountability = AccountabilityLogger()
        self.performance = PerformanceTracker(self.hierarchy)
        self.enabled = True

    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: list[AgentCapability],
        supervisor: BaseAgent | None = None
    ) -> None:
        """Register an agent with the responsibility system."""
        if not self.enabled:
            return

        self.hierarchy.add_agent(agent, capabilities)

        if supervisor:
            self.hierarchy.set_supervision_relationship(supervisor, agent)

        self.accountability.log_action(
            agent=agent,
            action_type="registration",
            action_description=f"Agent registered with {len(capabilities)} capabilities",
            context={"capabilities": [cap.name for cap in capabilities]}
        )

    def assign_task_responsibility(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        strategy: AssignmentStrategy = AssignmentStrategy.GREEDY,
        exclude_agents: list[BaseAgent] | None = None
    ) -> ResponsibilityAssignment | None:
        """Assign responsibility for a task to the best agent."""
        if not self.enabled:
            return None

        assignment = self.calculator.calculate_responsibility_assignment(
            task, requirements, strategy, exclude_agents
        )

        if assignment:
            agent = self._get_agent_by_id(assignment.agent_id)
            if agent:
                self.calculator.update_workload(agent, 1)

                self.accountability.log_action(
                    agent=agent,
                    action_type="task_assignment",
                    action_description=f"Assigned responsibility for task: {task.description[:100]}...",
                    task=task,
                    context={
                        "responsibility_score": assignment.responsibility_score,
                        "capability_matches": assignment.capability_matches,
                        "strategy": strategy.value
                    }
                )

        return assignment

    def complete_task(
        self,
        agent: BaseAgent,
        task: Task,
        success: bool,
        completion_time: float,
        quality_score: float | None = None,
        outcome_description: str = ""
    ) -> None:
        """Record task completion and update performance metrics."""
        if not self.enabled:
            return

        self.performance.record_task_completion(
            agent, success, completion_time, quality_score
        )

        self.calculator.update_workload(agent, -1)

        self.accountability.log_task_completion(
            agent, task, success, outcome_description, completion_time
        )

        adjustments = self.performance.adjust_capabilities_based_on_performance(agent)
        if adjustments:
            self.accountability.log_action(
                agent=agent,
                action_type="capability_adjustment",
                action_description="Capabilities adjusted based on performance",
                context={"adjustments": adjustments}
            )

    def delegate_task(
        self,
        delegating_agent: BaseAgent,
        receiving_agent: BaseAgent,
        task: Task,
        reason: str
    ) -> None:
        """Record task delegation between agents."""
        if not self.enabled:
            return

        self.calculator.update_workload(delegating_agent, -1)
        self.calculator.update_workload(receiving_agent, 1)

        self.accountability.log_delegation(
            delegating_agent, receiving_agent, task, reason
        )

    def get_agent_status(self, agent: BaseAgent) -> dict[str, Any]:
        """Get comprehensive status for an agent."""
        if not self.enabled:
            return {}

        agent_id = self.hierarchy._get_agent_id(agent)
        capabilities = self.hierarchy.get_agent_capabilities(agent)
        performance = self.performance.get_performance_metrics(agent)
        recent_records = self.accountability.get_agent_records(
            agent, since=datetime.utcnow() - timedelta(days=7)
        )
        current_workload = self.calculator.current_workloads.get(agent_id, 0)

        return {
            "agent_id": agent_id,
            "role": agent.role,
            "capabilities": [
                {
                    "name": cap.name,
                    "type": cap.capability_type.value,
                    "proficiency": cap.proficiency_level,
                    "confidence": cap.confidence_score
                }
                for cap in capabilities
            ],
            "performance": {
                "success_rate": performance.success_rate if performance else 0.0,
                "quality_score": performance.quality_score if performance else 0.0,
                "efficiency_score": performance.efficiency_score if performance else 0.0,
                "total_tasks": performance.total_tasks if performance else 0
            } if performance else None,
            "current_workload": current_workload,
            "recent_activity_count": len(recent_records)
        }

    def get_system_overview(self) -> dict[str, Any]:
        """Get overview of the entire responsibility system."""
        if not self.enabled:
            return {"enabled": False}

        total_agents = len(self.hierarchy.agents)
        capability_distribution = self.hierarchy.get_capability_distribution()
        workload_distribution = self.calculator.get_workload_distribution()

        all_performance = list(self.performance.performance_metrics.values())
        avg_success_rate = sum(p.success_rate for p in all_performance) / len(all_performance) if all_performance else 0.0
        avg_quality = sum(p.quality_score for p in all_performance) / len(all_performance) if all_performance else 0.0

        return {
            "enabled": True,
            "total_agents": total_agents,
            "capability_distribution": capability_distribution,
            "workload_distribution": workload_distribution,
            "system_performance": {
                "average_success_rate": avg_success_rate,
                "average_quality_score": avg_quality,
                "total_tasks_completed": sum(p.total_tasks for p in all_performance)
            },
            "total_accountability_records": len(self.accountability.records)
        }

    def generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate system-wide recommendations for improvement."""
        if not self.enabled:
            return []

        recommendations = []

        workloads = self.calculator.get_workload_distribution()
        if workloads:
            max_workload = max(workloads.values())
            min_workload = min(workloads.values())

            if max_workload - min_workload > 3:  # Significant imbalance
                recommendations.append({
                    "type": "workload_balancing",
                    "priority": "high",
                    "description": "Workload imbalance detected. Consider redistributing tasks.",
                    "details": {"max_workload": max_workload, "min_workload": min_workload}
                })

        capability_dist = self.hierarchy.get_capability_distribution()
        for cap_type, levels in capability_dist.items():
            total_agents_with_cap = sum(levels.values())
            if total_agents_with_cap < 2:  # Too few agents with this capability
                recommendations.append({
                    "type": "capability_gap",
                    "priority": "medium",
                    "description": f"Limited coverage for {cap_type.value} capabilities",
                    "details": {"capability_type": cap_type.value, "agent_count": total_agents_with_cap}
                })

        for agent_id, metrics in self.performance.performance_metrics.items():
            if metrics.success_rate < 0.6:  # Low success rate
                agent = self._get_agent_by_id(agent_id)
                if agent:
                    recommendations.append({
                        "type": "performance_improvement",
                        "priority": "high",
                        "description": f"Agent {agent.role} has low success rate",
                        "details": {
                            "agent_role": agent.role,
                            "success_rate": metrics.success_rate,
                            "improvement_opportunities": self.performance.identify_improvement_opportunities(agent)
                        }
                    })

        return recommendations

    def enable_system(self) -> None:
        """Enable the responsibility system."""
        self.enabled = True

    def disable_system(self) -> None:
        """Disable the responsibility system."""
        self.enabled = False

    def _get_agent_by_id(self, agent_id: str) -> BaseAgent | None:
        """Get agent by ID."""
        return self.hierarchy.agents.get(agent_id)

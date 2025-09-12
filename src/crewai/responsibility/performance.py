"""
Performance-based capability adjustment system.
"""

from datetime import timedelta
from typing import Any

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.hierarchy import CapabilityHierarchy
from crewai.responsibility.models import AgentCapability, PerformanceMetrics


class PerformanceTracker:
    """Tracks agent performance and adjusts capabilities accordingly."""

    def __init__(self, hierarchy: CapabilityHierarchy):
        self.hierarchy = hierarchy
        self.performance_metrics: dict[str, PerformanceMetrics] = {}
        self.learning_rate = 0.1
        self.adjustment_threshold = 0.05  # Minimum change to trigger capability update

    def record_task_completion(
        self,
        agent: BaseAgent,
        task_success: bool,
        completion_time: float,
        quality_score: float | None = None,
        capability_used: str | None = None
    ) -> None:
        """Record a task completion and update performance metrics."""
        agent_id = self._get_agent_id(agent)

        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = PerformanceMetrics(agent_id=agent_id)

        metrics = self.performance_metrics[agent_id]
        metrics.update_metrics(task_success, completion_time, quality_score)

        if capability_used and task_success is not None:
            self._update_capability_based_on_performance(
                agent, capability_used, task_success, quality_score
            )

    def get_performance_metrics(self, agent: BaseAgent) -> PerformanceMetrics | None:
        """Get performance metrics for an agent."""
        agent_id = self._get_agent_id(agent)
        return self.performance_metrics.get(agent_id)

    def adjust_capabilities_based_on_performance(
        self,
        agent: BaseAgent,
        performance_window: timedelta = timedelta(days=7)
    ) -> list[tuple[str, float, float]]:
        """Adjust agent capabilities based on recent performance."""
        agent_id = self._get_agent_id(agent)
        metrics = self.performance_metrics.get(agent_id)

        if not metrics:
            return []

        adjustments = []
        agent_capabilities = self.hierarchy.get_agent_capabilities(agent)

        for capability in agent_capabilities:
            old_proficiency = capability.proficiency_level
            old_confidence = capability.confidence_score

            new_proficiency, new_confidence = self._calculate_adjusted_capability(
                capability, metrics
            )

            proficiency_change = abs(new_proficiency - old_proficiency)
            confidence_change = abs(new_confidence - old_confidence)

            if proficiency_change >= self.adjustment_threshold or confidence_change >= self.adjustment_threshold:
                self.hierarchy.update_agent_capability(
                    agent, capability.name, new_proficiency, new_confidence
                )
                adjustments.append((capability.name, new_proficiency - old_proficiency, new_confidence - old_confidence))

        return adjustments

    def get_performance_trends(
        self,
        agent: BaseAgent,
        capability_name: str | None = None
    ) -> dict[str, list[float]]:
        """Get performance trends for an agent."""
        agent_id = self._get_agent_id(agent)
        metrics = self.performance_metrics.get(agent_id)

        if not metrics:
            return {}

        return {
            "success_rate": [metrics.success_rate],
            "quality_score": [metrics.quality_score],
            "efficiency_score": [metrics.efficiency_score],
            "reliability_score": [metrics.reliability_score]
        }

    def identify_improvement_opportunities(
        self,
        agent: BaseAgent
    ) -> list[dict[str, Any]]:
        """Identify areas where an agent could improve."""
        agent_id = self._get_agent_id(agent)
        metrics = self.performance_metrics.get(agent_id)

        if not metrics:
            return []

        opportunities = []

        if metrics.success_rate < 0.7:
            opportunities.append({
                "area": "success_rate",
                "current_value": metrics.success_rate,
                "recommendation": "Focus on task completion accuracy and problem-solving skills"
            })

        if metrics.quality_score < 0.6:
            opportunities.append({
                "area": "quality",
                "current_value": metrics.quality_score,
                "recommendation": "Improve attention to detail and output quality"
            })

        if metrics.efficiency_score < 0.5:
            opportunities.append({
                "area": "efficiency",
                "current_value": metrics.efficiency_score,
                "recommendation": "Work on time management and process optimization"
            })

        return opportunities

    def compare_agent_performance(
        self,
        agents: list[BaseAgent],
        metric: str = "overall"
    ) -> list[tuple[BaseAgent, float]]:
        """Compare performance across multiple agents."""
        agent_scores = []

        for agent in agents:
            agent_id = self._get_agent_id(agent)
            metrics = self.performance_metrics.get(agent_id)

            if not metrics:
                continue

            if metric == "overall":
                score = (
                    metrics.success_rate * 0.4 +
                    metrics.quality_score * 0.3 +
                    metrics.efficiency_score * 0.2 +
                    metrics.reliability_score * 0.1
                )
            elif metric == "success_rate":
                score = metrics.success_rate
            elif metric == "quality":
                score = metrics.quality_score
            elif metric == "efficiency":
                score = metrics.efficiency_score
            elif metric == "reliability":
                score = metrics.reliability_score
            else:
                continue

            agent_scores.append((agent, score))

        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores

    def _update_capability_based_on_performance(
        self,
        agent: BaseAgent,
        capability_name: str,
        task_success: bool,
        quality_score: float | None
    ) -> None:
        """Update a specific capability based on task performance."""
        agent_capabilities = self.hierarchy.get_agent_capabilities(agent)

        for capability in agent_capabilities:
            if capability.name == capability_name:
                if task_success:
                    proficiency_adjustment = self.learning_rate * 0.1  # Small positive adjustment
                    confidence_adjustment = self.learning_rate * 0.05
                else:
                    proficiency_adjustment = -self.learning_rate * 0.05  # Small negative adjustment
                    confidence_adjustment = -self.learning_rate * 0.1

                if quality_score is not None:
                    quality_factor = (quality_score - 0.5) * 2  # Scale to -1 to 1
                    proficiency_adjustment *= (1 + quality_factor * 0.5)

                new_proficiency = max(0.0, min(1.0, capability.proficiency_level + proficiency_adjustment))
                new_confidence = max(0.0, min(1.0, capability.confidence_score + confidence_adjustment))

                self.hierarchy.update_agent_capability(
                    agent, capability_name, new_proficiency, new_confidence
                )
                break

    def _calculate_adjusted_capability(
        self,
        capability: AgentCapability,
        metrics: PerformanceMetrics
    ) -> tuple[float, float]:
        """Calculate adjusted capability values based on performance metrics."""
        performance_factor = (
            metrics.success_rate * 0.4 +
            metrics.quality_score * 0.3 +
            metrics.efficiency_score * 0.2 +
            metrics.reliability_score * 0.1
        )

        adjustment_magnitude = (performance_factor - 0.5) * self.learning_rate

        new_proficiency = capability.proficiency_level + adjustment_magnitude
        new_proficiency = max(0.0, min(1.0, new_proficiency))

        confidence_adjustment = (metrics.reliability_score - 0.5) * self.learning_rate * 0.5
        new_confidence = capability.confidence_score + confidence_adjustment
        new_confidence = max(0.0, min(1.0, new_confidence))

        return new_proficiency, new_confidence

    def _get_agent_id(self, agent: BaseAgent) -> str:
        """Get a unique identifier for an agent."""
        return f"{agent.role}_{id(agent)}"

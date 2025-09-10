"""
Mathematical responsibility assignment algorithms.
"""

import math
from enum import Enum

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.hierarchy import CapabilityHierarchy
from crewai.responsibility.models import ResponsibilityAssignment, TaskRequirement
from crewai.task import Task


class AssignmentStrategy(str, Enum):
    """Different strategies for responsibility assignment."""
    GREEDY = "greedy"  # Assign to best available agent
    BALANCED = "balanced"  # Balance workload across agents
    OPTIMAL = "optimal"  # Optimize for overall system performance


class ResponsibilityCalculator:
    """Calculates and assigns responsibilities using mathematical algorithms."""

    def __init__(self, hierarchy: CapabilityHierarchy):
        self.hierarchy = hierarchy
        self.current_workloads: dict[str, int] = {}  # agent_id -> current task count

    def calculate_responsibility_assignment(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        strategy: AssignmentStrategy = AssignmentStrategy.GREEDY,
        exclude_agents: list[BaseAgent] | None = None
    ) -> ResponsibilityAssignment | None:
        """Calculate the best responsibility assignment for a task."""
        exclude_agent_ids = set()
        if exclude_agents:
            exclude_agent_ids = {self.hierarchy._get_agent_id(agent) for agent in exclude_agents}

        if strategy == AssignmentStrategy.GREEDY:
            return self._greedy_assignment(task, requirements, exclude_agent_ids)
        if strategy == AssignmentStrategy.BALANCED:
            return self._balanced_assignment(task, requirements, exclude_agent_ids)
        if strategy == AssignmentStrategy.OPTIMAL:
            return self._optimal_assignment(task, requirements, exclude_agent_ids)
        raise ValueError(f"Unknown assignment strategy: {strategy}")

    def calculate_multi_agent_assignment(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        max_agents: int = 3,
        strategy: AssignmentStrategy = AssignmentStrategy.OPTIMAL
    ) -> list[ResponsibilityAssignment]:
        """Calculate assignment for tasks requiring multiple agents."""
        assignments = []
        used_agents = set()

        sorted_requirements = sorted(requirements, key=lambda r: r.weight, reverse=True)

        for i, requirement in enumerate(sorted_requirements):
            if len(assignments) >= max_agents:
                break

            single_req_assignment = self.calculate_responsibility_assignment(
                task, [requirement], strategy,
                exclude_agents=[self.hierarchy.agents[agent_id] for agent_id in used_agents]
            )

            if single_req_assignment:
                single_req_assignment.responsibility_score *= (1.0 / (i + 1))  # Diminishing returns
                assignments.append(single_req_assignment)
                used_agents.add(single_req_assignment.agent_id)

        return assignments

    def update_workload(self, agent: BaseAgent, workload_change: int) -> None:
        """Update the current workload for an agent."""
        agent_id = self.hierarchy._get_agent_id(agent)
        current = self.current_workloads.get(agent_id, 0)
        self.current_workloads[agent_id] = max(0, current + workload_change)

    def get_workload_distribution(self) -> dict[str, int]:
        """Get current workload distribution across all agents."""
        return self.current_workloads.copy()

    def _greedy_assignment(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        exclude_agent_ids: set
    ) -> ResponsibilityAssignment | None:
        """Assign to the agent with highest capability match score."""
        best_match = self.hierarchy.get_best_agent_for_task(requirements, exclude_agent_ids)

        if not best_match:
            return None

        agent, score, matches = best_match
        agent_id = self.hierarchy._get_agent_id(agent)

        return ResponsibilityAssignment(
            agent_id=agent_id,
            task_id=str(task.id),
            responsibility_score=score,
            capability_matches=matches,
            reasoning=f"Greedy assignment: highest capability match score ({score:.3f})"
        )

    def _balanced_assignment(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        exclude_agent_ids: set
    ) -> ResponsibilityAssignment | None:
        """Assign considering both capability and current workload."""
        capable_agents = self.hierarchy.find_capable_agents(requirements, minimum_match_score=0.3)

        if not capable_agents:
            return None

        best_agent = None
        best_score = -1.0
        best_matches = []

        for agent, capability_score in capable_agents:
            agent_id = self.hierarchy._get_agent_id(agent)

            if agent_id in exclude_agent_ids:
                continue

            current_workload = self.current_workloads.get(agent_id, 0)
            workload_penalty = self._calculate_workload_penalty(current_workload)

            combined_score = capability_score * (1.0 - workload_penalty)

            if combined_score > best_score:
                best_score = combined_score
                best_agent = agent
                _, best_matches = self.hierarchy._calculate_detailed_capability_match(agent_id, requirements)

        if best_agent:
            agent_id = self.hierarchy._get_agent_id(best_agent)
            return ResponsibilityAssignment(
                agent_id=agent_id,
                task_id=str(task.id),
                responsibility_score=best_score,
                capability_matches=best_matches,
                reasoning=f"Balanced assignment: capability ({capability_score:.3f}) with workload consideration"
            )

        return None

    def _optimal_assignment(
        self,
        task: Task,
        requirements: list[TaskRequirement],
        exclude_agent_ids: set
    ) -> ResponsibilityAssignment | None:
        """Assign using optimization for overall system performance."""
        capable_agents = self.hierarchy.find_capable_agents(requirements, minimum_match_score=0.2)

        if not capable_agents:
            return None

        best_agent = None
        best_score = -1.0
        best_matches = []

        for agent, capability_score in capable_agents:
            agent_id = self.hierarchy._get_agent_id(agent)

            if agent_id in exclude_agent_ids:
                continue

            optimization_score = self._calculate_optimization_score(
                agent_id, capability_score, requirements
            )

            if optimization_score > best_score:
                best_score = optimization_score
                best_agent = agent
                _, best_matches = self.hierarchy._calculate_detailed_capability_match(agent_id, requirements)

        if best_agent:
            agent_id = self.hierarchy._get_agent_id(best_agent)
            return ResponsibilityAssignment(
                agent_id=agent_id,
                task_id=str(task.id),
                responsibility_score=best_score,
                capability_matches=best_matches,
                reasoning=f"Optimal assignment: multi-factor optimization score ({best_score:.3f})"
            )

        return None

    def _calculate_workload_penalty(self, current_workload: int) -> float:
        """Calculate penalty based on current workload."""
        if current_workload == 0:
            return 0.0

        return min(0.8, 1.0 - math.exp(-current_workload / 3.0))

    def _calculate_optimization_score(
        self,
        agent_id: str,
        capability_score: float,
        requirements: list[TaskRequirement]
    ) -> float:
        """Calculate multi-factor optimization score."""
        score = capability_score

        current_workload = self.current_workloads.get(agent_id, 0)
        workload_factor = 1.0 - self._calculate_workload_penalty(current_workload)

        agent_capabilities = self.hierarchy.agent_capabilities.get(agent_id, [])
        specialization_bonus = self._calculate_specialization_bonus(agent_capabilities, requirements)

        reliability_factor = 1.0  # Placeholder for future performance integration

        return (
            score * 0.5 +  # 50% capability match
            workload_factor * 0.2 +  # 20% workload consideration
            specialization_bonus * 0.2 +  # 20% specialization bonus
            reliability_factor * 0.1  # 10% reliability
        )

    def _calculate_specialization_bonus(
        self,
        agent_capabilities: list,
        requirements: list[TaskRequirement]
    ) -> float:
        """Calculate bonus for agents with specialized capabilities."""
        if not agent_capabilities or not requirements:
            return 0.0

        high_proficiency_matches = 0
        total_matches = 0

        for capability in agent_capabilities:
            for requirement in requirements:
                if self.hierarchy._capabilities_match(capability, requirement):
                    total_matches += 1
                    if capability.proficiency_level >= 0.8:
                        high_proficiency_matches += 1

        if total_matches == 0:
            return 0.0

        specialization_ratio = high_proficiency_matches / total_matches
        return min(0.3, specialization_ratio * 0.3)  # Max 30% bonus

"""
Capability-based agent hierarchy management.
"""

from collections import defaultdict, deque

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.responsibility.models import (
    AgentCapability,
    CapabilityType,
    TaskRequirement,
)


class CapabilityHierarchy:
    """Manages capability-based agent hierarchy and relationships."""

    def __init__(self):
        self.agents: dict[str, BaseAgent] = {}
        self.agent_capabilities: dict[str, list[AgentCapability]] = defaultdict(list)
        self.capability_index: dict[str, set[str]] = defaultdict(set)  # capability_name -> agent_ids
        self.hierarchy_relationships: dict[str, set[str]] = defaultdict(set)  # supervisor -> subordinates

    def add_agent(self, agent: BaseAgent, capabilities: list[AgentCapability]) -> None:
        """Add an agent with their capabilities to the hierarchy."""
        agent_id = self._get_agent_id(agent)
        self.agents[agent_id] = agent
        self.agent_capabilities[agent_id] = capabilities

        for capability in capabilities:
            self.capability_index[capability.name].add(agent_id)

    def remove_agent(self, agent: BaseAgent) -> None:
        """Remove an agent from the hierarchy."""
        agent_id = self._get_agent_id(agent)

        if agent_id in self.agents:
            for capability in self.agent_capabilities[agent_id]:
                self.capability_index[capability.name].discard(agent_id)

            for supervisor_id in self.hierarchy_relationships:
                self.hierarchy_relationships[supervisor_id].discard(agent_id)
            if agent_id in self.hierarchy_relationships:
                del self.hierarchy_relationships[agent_id]

            del self.agents[agent_id]
            del self.agent_capabilities[agent_id]

    def set_supervision_relationship(self, supervisor: BaseAgent, subordinate: BaseAgent) -> None:
        """Establish a supervision relationship between agents."""
        supervisor_id = self._get_agent_id(supervisor)
        subordinate_id = self._get_agent_id(subordinate)

        if supervisor_id in self.agents and subordinate_id in self.agents:
            self.hierarchy_relationships[supervisor_id].add(subordinate_id)

    def get_agent_capabilities(self, agent: BaseAgent) -> list[AgentCapability]:
        """Get capabilities for a specific agent."""
        agent_id = self._get_agent_id(agent)
        return self.agent_capabilities.get(agent_id, [])

    def update_agent_capability(
        self,
        agent: BaseAgent,
        capability_name: str,
        new_proficiency: float,
        new_confidence: float
    ) -> bool:
        """Update a specific capability for an agent."""
        agent_id = self._get_agent_id(agent)

        if agent_id not in self.agent_capabilities:
            return False

        for capability in self.agent_capabilities[agent_id]:
            if capability.name == capability_name:
                capability.update_proficiency(new_proficiency, new_confidence)
                return True

        return False

    def find_capable_agents(
        self,
        requirements: list[TaskRequirement],
        minimum_match_score: float = 0.5
    ) -> list[tuple[BaseAgent, float]]:
        """Find agents capable of handling the given requirements."""
        agent_scores = []

        for agent_id, agent in self.agents.items():
            score = self._calculate_capability_match_score(agent_id, requirements)
            if score >= minimum_match_score:
                agent_scores.append((agent, score))

        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return agent_scores

    def get_best_agent_for_task(
        self,
        requirements: list[TaskRequirement],
        exclude_agents: set[str] | None = None
    ) -> tuple[BaseAgent, float, list[str]] | None:
        """Get the best agent for a task based on capability requirements."""
        exclude_agents = exclude_agents or set()
        best_agent = None
        best_score = 0.0
        best_matches = []

        for agent_id, agent in self.agents.items():
            if agent_id in exclude_agents:
                continue

            score, matches = self._calculate_detailed_capability_match(agent_id, requirements)
            if score > best_score:
                best_score = score
                best_agent = agent
                best_matches = matches

        if best_agent:
            return best_agent, best_score, best_matches
        return None

    def get_subordinates(self, supervisor: BaseAgent) -> list[BaseAgent]:
        """Get all subordinates of a supervisor agent."""
        supervisor_id = self._get_agent_id(supervisor)
        subordinate_ids = self.hierarchy_relationships.get(supervisor_id, set())
        return [self.agents[sub_id] for sub_id in subordinate_ids if sub_id in self.agents]

    def get_hierarchy_path(self, from_agent: BaseAgent, to_agent: BaseAgent) -> list[BaseAgent] | None:
        """Find the shortest path in the hierarchy between two agents."""
        from_id = self._get_agent_id(from_agent)
        to_id = self._get_agent_id(to_agent)

        if from_id not in self.agents or to_id not in self.agents:
            return None

        queue = deque([(from_id, [from_id])])
        visited = {from_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == to_id:
                return [self.agents[agent_id] for agent_id in path]

            for subordinate_id in self.hierarchy_relationships.get(current_id, set()):
                if subordinate_id not in visited:
                    visited.add(subordinate_id)
                    queue.append((subordinate_id, [*path, subordinate_id]))

        return None

    def get_capability_distribution(self) -> dict[CapabilityType, dict[str, int]]:
        """Get distribution of capabilities across all agents."""
        distribution = defaultdict(lambda: defaultdict(int))

        for capabilities in self.agent_capabilities.values():
            for capability in capabilities:
                proficiency_level = "high" if capability.proficiency_level >= 0.8 else \
                                 "medium" if capability.proficiency_level >= 0.5 else "low"
                distribution[capability.capability_type][proficiency_level] += 1

        return dict(distribution)

    def _get_agent_id(self, agent: BaseAgent) -> str:
        """Get a unique identifier for an agent."""
        return f"{agent.role}_{id(agent)}"

    def _calculate_capability_match_score(
        self,
        agent_id: str,
        requirements: list[TaskRequirement]
    ) -> float:
        """Calculate how well an agent's capabilities match task requirements."""
        if not requirements:
            return 1.0

        agent_capabilities = self.agent_capabilities.get(agent_id, [])
        if not agent_capabilities:
            return 0.0

        total_weight = sum(req.weight for req in requirements)
        if total_weight == 0:
            return 0.0

        weighted_score = 0.0

        for requirement in requirements:
            best_match_score = 0.0

            for capability in agent_capabilities:
                if self._capabilities_match(capability, requirement):
                    proficiency_score = min(capability.proficiency_level / requirement.minimum_proficiency, 1.0)
                    confidence_factor = capability.confidence_score
                    match_score = proficiency_score * confidence_factor
                    best_match_score = max(best_match_score, match_score)

            weighted_score += best_match_score * requirement.weight

        return weighted_score / total_weight

    def _calculate_detailed_capability_match(
        self,
        agent_id: str,
        requirements: list[TaskRequirement]
    ) -> tuple[float, list[str]]:
        """Calculate detailed capability match with matched capability names."""
        if not requirements:
            return 1.0, []

        agent_capabilities = self.agent_capabilities.get(agent_id, [])
        if not agent_capabilities:
            return 0.0, []

        total_weight = sum(req.weight for req in requirements)
        if total_weight == 0:
            return 0.0, []

        weighted_score = 0.0
        matched_capabilities = []

        for requirement in requirements:
            best_match_score = 0.0
            best_match_capability = None

            for capability in agent_capabilities:
                if self._capabilities_match(capability, requirement):
                    proficiency_score = min(capability.proficiency_level / requirement.minimum_proficiency, 1.0)
                    confidence_factor = capability.confidence_score
                    match_score = proficiency_score * confidence_factor

                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_capability = capability.name

            if best_match_capability:
                matched_capabilities.append(best_match_capability)

            weighted_score += best_match_score * requirement.weight

        return weighted_score / total_weight, matched_capabilities

    def _capabilities_match(self, capability: AgentCapability, requirement: TaskRequirement) -> bool:
        """Check if a capability matches a requirement."""
        if capability.name.lower() == requirement.capability_name.lower():
            return True

        if capability.capability_type == requirement.capability_type:
            return True

        capability_keywords = set(kw.lower() for kw in capability.keywords)
        requirement_keywords = set(kw.lower() for kw in requirement.keywords)

        if capability_keywords.intersection(requirement_keywords):
            return True

        return False

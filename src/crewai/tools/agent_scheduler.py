import random
from typing import Dict, List


class AgentScheduler:
    """
    Tracks agent performance and suggests dynamic retraining intervals.
    """

    def __init__(self, agent_ids: List[str]):
        self.performance_log: Dict[str, List[float]] = {
            agent_id: [] for agent_id in agent_ids
        }

    def track_performance(self, agent_id: str, success: bool):
        self.performance_log[agent_id].append(1.0 if success else 0.0)

    def adjust_training_schedule(self, agent_id: str) -> int:
        log = self.performance_log.get(agent_id, [])
        if not log:
            return 3  # Default if no data

        avg_score = sum(log[-10:]) / min(len(log), 10)
        if avg_score < 0.5:
            return 1  # Frequent retraining
        elif avg_score > 0.8:
            return 5  # Rare retraining
        return 3  # Moderate


# Optional test harness
if __name__ == "__main__":
    agents = ["agent_alpha", "agent_beta", "agent_gamma"]
    scheduler = AgentScheduler(agent_ids=agents)

    for _ in range(10):
        for agent in agents:
            result = random.choice([True, False])
            scheduler.track_performance(agent, result)

    for agent in agents:
        interval = scheduler.adjust_training_schedule(agent)
        print(f"{agent} → retrain every {interval} days")

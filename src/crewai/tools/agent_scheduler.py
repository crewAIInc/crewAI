import random
from typing import Dict, List
from crewai.tools import BaseTool
from pydantic import Field

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


class AgentSchedulerTool(BaseTool):
    name: str = "agent_scheduler"
    description: str = (
        "Tracks agent performance and suggests dynamic retraining intervals. "
        "Takes agent_id (e.g., 'agent_alpha') and performance (comma-separated values like 'True,False,True')"
    )
    agent_ids: List[str]
    scheduler: AgentScheduler = Field(default=None)

    def __init__(self, agent_ids: List[str]):
        super().__init__(agent_ids=agent_ids)
        object.__setattr__(self, 'scheduler', AgentScheduler(agent_ids))

    def _run(self, agent_id: str, performance: str) -> str:
        try:
            performance_list = [x.strip() == "True" for x in performance.split(",")]
            for result in performance_list:
                self.scheduler.track_performance(agent_id, result)
            interval = self.scheduler.adjust_training_schedule(agent_id)
            return f"Recommended retraining interval for {agent_id}: {interval} days"
        except Exception as e:
            return f"Error processing input: {e}"

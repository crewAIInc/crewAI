from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ExecutionStep(BaseModel):
    """Represents a single step in the crew execution trace."""
    
    timestamp: datetime = Field(description="When this step occurred")
    step_type: str = Field(description="Type of step: agent_thought, tool_call, tool_result, task_start, task_complete, etc.")
    agent_role: Optional[str] = Field(description="Role of the agent performing this step", default=None)
    task_description: Optional[str] = Field(description="Description of the task being executed", default=None)
    content: Dict[str, Any] = Field(description="Step-specific content (thought, tool args, result, etc.)", default_factory=dict)
    metadata: Dict[str, Any] = Field(description="Additional metadata for this step", default_factory=dict)

class ExecutionTrace(BaseModel):
    """Complete execution trace for a crew run."""
    
    steps: List[ExecutionStep] = Field(description="Ordered list of execution steps", default_factory=list)
    total_steps: int = Field(description="Total number of steps in the trace", default=0)
    start_time: Optional[datetime] = Field(description="When execution started", default=None)
    end_time: Optional[datetime] = Field(description="When execution completed", default=None)
    
    def add_step(self, step: ExecutionStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)
        self.total_steps = len(self.steps)
    
    def get_steps_by_type(self, step_type: str) -> List[ExecutionStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]
    
    def get_steps_by_agent(self, agent_role: str) -> List[ExecutionStep]:
        """Get all steps performed by a specific agent."""
        return [step for step in self.steps if step.agent_role == agent_role]

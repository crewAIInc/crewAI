"""
Data models for the formal responsibility tracking system.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CapabilityType(str, Enum):
    """Types of capabilities an agent can have."""
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    COMMUNICATION = "communication"
    LEADERSHIP = "leadership"
    DOMAIN_SPECIFIC = "domain_specific"


class AgentCapability(BaseModel):
    """Represents a specific capability of an agent."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Name of the capability")
    capability_type: CapabilityType = Field(..., description="Type of capability")
    proficiency_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proficiency level from 0.0 to 1.0"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this capability assessment"
    )
    description: str | None = Field(None, description="Detailed description of the capability")
    keywords: list[str] = Field(default_factory=list, description="Keywords associated with this capability")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def update_proficiency(self, new_level: float, confidence: float) -> None:
        """Update proficiency level and confidence."""
        self.proficiency_level = max(0.0, min(1.0, new_level))
        self.confidence_score = max(0.0, min(1.0, confidence))
        self.last_updated = datetime.utcnow()


class ResponsibilityAssignment(BaseModel):
    """Represents the assignment of responsibility for a task to an agent."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(..., description="ID of the assigned agent")
    task_id: str = Field(..., description="ID of the task")
    responsibility_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calculated responsibility score"
    )
    capability_matches: list[str] = Field(
        default_factory=list,
        description="Capabilities that matched for this assignment"
    )
    reasoning: str = Field(..., description="Explanation for this assignment")
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(None)
    success: bool | None = Field(None, description="Whether the assignment was successful")

    def mark_completed(self, success: bool) -> None:
        """Mark the assignment as completed."""
        self.completed_at = datetime.utcnow()
        self.success = success


class AccountabilityRecord(BaseModel):
    """Records agent actions and decisions for accountability tracking."""

    id: UUID = Field(default_factory=uuid4)
    agent_id: str = Field(..., description="ID of the agent")
    action_type: str = Field(..., description="Type of action taken")
    action_description: str = Field(..., description="Description of the action")
    task_id: str | None = Field(None, description="Related task ID if applicable")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    outcome: str | None = Field(None, description="Outcome of the action")
    success: bool | None = Field(None, description="Whether the action was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def set_outcome(self, outcome: str, success: bool) -> None:
        """Set the outcome of the action."""
        self.outcome = outcome
        self.success = success


class PerformanceMetrics(BaseModel):
    """Performance metrics for an agent."""

    agent_id: str = Field(..., description="ID of the agent")
    total_tasks: int = Field(default=0, description="Total number of tasks assigned")
    successful_tasks: int = Field(default=0, description="Number of successful tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    average_completion_time: float = Field(default=0.0, description="Average task completion time in seconds")
    quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall quality score"
    )
    efficiency_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Efficiency score based on completion times"
    )
    reliability_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Reliability score based on success rate"
    )
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    def update_metrics(
        self,
        task_success: bool,
        completion_time: float,
        quality_score: float | None = None
    ) -> None:
        """Update performance metrics with new task result."""
        self.total_tasks += 1
        if task_success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1

        alpha = 0.1  # Learning rate

        if self.total_tasks == 1:
            self.average_completion_time = completion_time
        else:
            self.average_completion_time = (
                alpha * completion_time + (1 - alpha) * self.average_completion_time
            )

        self.reliability_score = self.success_rate

        if completion_time > 0:
            normalized_time = min(completion_time / 3600, 1.0)  # Normalize to hours, cap at 1
            self.efficiency_score = max(0.1, 1.0 - normalized_time)

        if quality_score is not None:
            self.quality_score = (
                alpha * quality_score + (1 - alpha) * self.quality_score
            )

        self.last_updated = datetime.utcnow()


class TaskRequirement(BaseModel):
    """Represents capability requirements for a task."""

    capability_name: str = Field(..., description="Name of required capability")
    capability_type: CapabilityType = Field(..., description="Type of required capability")
    minimum_proficiency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum required proficiency level"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight/importance of this requirement"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that help match capabilities"
    )

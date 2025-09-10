"""Human-in-the-loop (HITL) type definitions.

This module provides type definitions for human-in-the-loop interactions
in crew executions.
"""

from typing import TypedDict


class HITLResumeInfo(TypedDict, total=False):
    """HITL resume information passed from flow to crew.

    Attributes:
        task_id: Unique identifier for the task.
        crew_execution_id: Unique identifier for the crew execution.
        task_key: Key identifying the specific task.
        task_output: Output from the task before human intervention.
        human_feedback: Feedback provided by the human.
        previous_messages: History of messages in the conversation.
    """

    task_id: str
    crew_execution_id: str
    task_key: str
    task_output: str
    human_feedback: str
    previous_messages: list[dict[str, str]]


class CrewInputsWithHITL(TypedDict, total=False):
    """Crew inputs that may contain HITL resume information.

    Attributes:
        _hitl_resume: Optional HITL resume information for continuing execution.
    """

    _hitl_resume: HITLResumeInfo

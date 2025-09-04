from typing import TypedDict


class HITLResumeInfo(TypedDict, total=False):
    """HITL resume information passed from flow to crew."""

    task_id: str
    crew_execution_id: str
    task_key: str
    task_output: str
    human_feedback: str
    previous_messages: list[dict[str, str]]


class CrewInputsWithHITL(TypedDict, total=False):
    """Crew inputs that may contain HITL resume information."""

    _hitl_resume: HITLResumeInfo

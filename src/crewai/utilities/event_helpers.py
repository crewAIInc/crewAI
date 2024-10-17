# event_helpers.py

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from crewai.utilities.event_emitter import CrewEvents, emit

if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.crews.crew_output import CrewOutput
    from crewai.task import Task
    from crewai.tasks.task_output import TaskOutput


def emit_crew_start(
    crew: "Crew",  # Use a forward reference
    inputs: Optional[Dict[str, Any]] = None,
) -> None:
    serialized_crew = crew.serialize()
    emit(
        CrewEvents.CREW_START,
        {
            **serialized_crew,
        },
        inputs=inputs,
    )


def emit_crew_finish(crew: "Crew", result: "CrewOutput") -> None:
    serialized_crew = crew.serialize()
    serialized_result = result.serialize()
    print("emit crew finish")

    emit(
        CrewEvents.CREW_FINISH,
        {
            **serialized_crew,
            "result": serialized_result,
        },
    )


def emit_crew_failure(
    crew_id: str, name: str, error: Exception, traceback: str, duration: float
) -> None:
    emit(
        CrewEvents.CREW_FAILURE,
        {
            "crew_id": crew_id,
            "name": name,
            "failure_time": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback,
            "duration": duration,
        },
    )


def emit_task_start(
    task: "Task",
    agent_role: str = "None",
) -> None:
    serialized_task = task.serialize()
    emit(
        CrewEvents.TASK_START,
        {
            **serialized_task,
        },
        agent_role=agent_role,
    )


def emit_task_finish(
    task: "Task",
    inputs: Dict[str, Any],
    output: "TaskOutput",
    task_index: int,
    was_replayed: bool = False,
) -> None:
    emit(
        CrewEvents.TASK_FINISH,
        {
            "task": task.serialize(),
            "output": {
                "description": output.description,
                "summary": output.summary,
                "raw": output.raw,
                "pydantic": output.pydantic,
                "json_dict": output.json_dict,
                "output_format": output.output_format,
                "agent": output.agent,
            },
            "task_index": task_index,
            "inputs": inputs,
            "was_replayed": was_replayed,
        },
    )


def emit_task_failure(
    crew_id: str,
    task_id: str,
    task_name: str,
    error: Exception,
    traceback: str,
    duration: float,
) -> None:
    emit(
        CrewEvents.TASK_FAILURE,
        {
            "crew_id": crew_id,
            "task_id": task_id,
            "task_name": task_name,
            "failure_time": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback,
            "duration": duration,
        },
    )

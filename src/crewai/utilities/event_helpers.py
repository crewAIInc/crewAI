# event_helpers.py

from datetime import datetime
from typing import Any, Dict, Optional

from crewai.crew import Crew
from crewai.utilities.event_emitter import CrewEvents, emit


def emit_crew_start(
    crew: Crew,
    inputs: Optional[Dict[str, Any]] = None,
) -> None:
    serialized_crew = crew.serialize()
    emit(
        CrewEvents.CREW_START,
        {**serialized_crew, "inputs": inputs},
    )


def emit_crew_finish(crew_id: str, name: str, result: Any, duration: float) -> None:
    emit(
        CrewEvents.CREW_FINISH,
        {
            "crew_id": crew_id,
            "name": name,
            "finish_time": datetime.now().isoformat(),
            "result": result,
            "duration": duration,
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


def emit_task_start(crew_id: str, task_id: str, task_name: str) -> None:
    emit(
        CrewEvents.TASK_START,
        {
            "crew_id": crew_id,
            "task_id": task_id,
            "task_name": task_name,
            "start_time": datetime.now().isoformat(),
        },
    )


def emit_task_finish(
    crew_id: str, task_id: str, task_name: str, result: Any, duration: float
) -> None:
    emit(
        CrewEvents.TASK_FINISH,
        {
            "crew_id": crew_id,
            "task_id": task_id,
            "task_name": task_name,
            "finish_time": datetime.now().isoformat(),
            "result": result,
            "duration": duration,
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

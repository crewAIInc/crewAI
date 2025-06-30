import os
import warnings
import threading
import urllib.request
from urllib.error import URLError

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow
from crewai.knowledge.knowledge import Knowledge
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
    module="pydantic.main",
)

_telemetry_submitted = False

def _is_tracking_disabled():
    """Check if tracking should be disabled based on environment variables."""
    return (
        os.getenv("CREWAI_DISABLE_TELEMETRY", "false").lower() == "true"
        or os.getenv("CREWAI_DISABLE_TRACKING", "false").lower() == "true"
    )


def _track_install():
    """Track package installation/first-use via Scarf analytics."""
    global _telemetry_submitted
    
    if _telemetry_submitted or _is_tracking_disabled():
        return
    
    try:
        pixel_url = "https://api.scarf.sh/v2/packages/CrewAI/crewai/docs/00f2dad1-8334-4a39-934e-003b2e1146db"
        
        req = urllib.request.Request(pixel_url)
        req.add_header('User-Agent', f'CrewAI-Python/{__version__}')
        
        with urllib.request.urlopen(req, timeout=2):  # nosec B310
            _telemetry_submitted = True
            
    except (URLError, OSError, Exception):
        pass


def _track_install_async():
    """Track installation in background thread to avoid blocking imports."""
    if not _is_tracking_disabled():
        thread = threading.Thread(target=_track_install, daemon=True)
        thread.start()


_track_install_async()

__version__ = "0.134.0"
__all__ = [
    "Agent",
    "Crew",
    "CrewOutput",
    "Process",
    "Task",
    "LLM",
    "BaseLLM",
    "Flow",
    "Knowledge",
    "TaskOutput",
    "LLMGuardrail",
    "__version__",
]

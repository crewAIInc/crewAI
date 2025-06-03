"""MLflow integration for CrewAI"""
import logging

from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.third_party.mlflow_listener import mlflow_listener

logger = logging.getLogger(__name__)


def autolog(
    disable: bool = False,
    silent: bool = False,
) -> None:
    """
    Enable or disable MLflow autologging for CrewAI.
    
    Args:
        disable: If True, disable autologging. If False, enable it.
        silent: If True, suppress logging messages.
        
    Raises:
        TypeError: If disable or silent are not boolean values.
    """
    if not isinstance(disable, bool) or not isinstance(silent, bool):
        raise TypeError("Parameters 'disable' and 'silent' must be boolean")
    try:
        import mlflow  # noqa: F401
    except ImportError:
        if not silent:
            logger.warning(
                "MLflow is not installed. Install it with: pip install mlflow>=2.19.0"
            )
        return
    
    if disable:
        mlflow_listener._autolog_enabled = False
        if not silent:
            logger.info("MLflow autologging disabled for CrewAI")
    else:
        mlflow_listener.setup_listeners(crewai_event_bus)
        mlflow_listener._autolog_enabled = True
        if not silent:
            logger.info("MLflow autologging enabled for CrewAI")


def _patch_mlflow():
    """Patch MLflow to include crewai.autolog()"""
    try:
        import mlflow
        if not hasattr(mlflow, 'crewai'):
            class CrewAIModule:
                autolog = staticmethod(autolog)
            
            mlflow.crewai = CrewAIModule()
    except ImportError:
        pass


_patch_mlflow()

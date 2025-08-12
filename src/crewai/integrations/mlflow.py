"""MLFlow integration utilities for CrewAI."""

import logging

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Check if MLFlow is available."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def setup_mlflow_autolog(
    log_traces: bool = True,
    log_models: bool = False,
    disable: bool = False,
    exclusive: bool = False,
    disable_for_unsupported_versions: bool = False,
    silent: bool = False,
) -> bool:
    """
    Setup MLFlow autologging for CrewAI.
    
    This is a convenience wrapper around mlflow.crewai.autolog() that provides
    better error handling and documentation.
    
    Args:
        log_traces: Whether to log traces
        log_models: Whether to log models
        disable: Whether to disable autologging
        exclusive: Whether to use exclusive mode
        disable_for_unsupported_versions: Whether to disable for unsupported versions
        silent: Whether to suppress warnings
        
    Returns:
        True if autologging was successfully enabled, False otherwise
    """
    if not is_mlflow_available():
        if not silent:
            logger.warning(
                "MLFlow is not available. Install it with: pip install mlflow"
            )
        return False
    
    try:
        import mlflow
        mlflow.crewai.autolog(
            log_traces=log_traces,
            log_models=log_models,
            disable=disable,
            exclusive=exclusive,
            disable_for_unsupported_versions=disable_for_unsupported_versions,
            silent=silent,
        )
        if not silent:
            logger.info("MLFlow autologging enabled for CrewAI")
        return True
    except Exception as e:
        if not silent:
            logger.error(f"Failed to enable MLFlow autologging: {e}")
        return False


def get_active_run():
    """Get the active MLFlow run if available."""
    if not is_mlflow_available():
        return None
    
    try:
        import mlflow
        return mlflow.active_run()
    except Exception:
        return None


def log_crew_execution(crew_name: str, **kwargs):
    """Log crew execution details to MLFlow if available."""
    if not is_mlflow_available():
        return
    
    try:
        import mlflow
        with mlflow.start_run(run_name=f"crew_{crew_name}"):
            for key, value in kwargs.items():
                mlflow.log_param(key, value)
    except Exception as e:
        logger.debug(f"Failed to log crew execution to MLFlow: {e}")

"""Logging and warning utility functions for CrewAI."""

from collections.abc import Generator
import contextlib
import io
import logging
import os
import warnings


@contextlib.contextmanager
def suppress_logging(
    logger_name: str,
    level: int | str,
) -> Generator[None, None, None]:
    """Suppress verbose logging output from specified logger.

    Commonly used to suppress ChromaDB's verbose HNSW index logging.

    Args:
        logger_name: The logger to suppress
        level: The minimum level to allow (e.g., logging.ERROR or "ERROR")

    Yields:
        None

    Example:
        with suppress_logging("chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR):
            collection.query(query_texts=["test"])
    """
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)


@contextlib.contextmanager
def suppress_warnings() -> Generator[None, None, None]:
    """Context manager to suppress all warnings.

    Yields:
        None during the context execution.

    Note:
        This implementation consolidates warning suppression used throughout
        the codebase, including specific deprecation warnings from dependencies.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        warnings.filterwarnings(
            "ignore", message="open_text is deprecated*", category=DeprecationWarning
        )
        yield


def should_enable_verbose(*, override: bool | None = None) -> bool:
    """Determine if verbose logging should be enabled.

    This is the single source of truth for verbose logging enablement.
    Priority order:
    1. Explicit override (e.g., Crew.verbose=True/False or Flow.verbose=True/False)
    2. Environment variable CREWAI_VERBOSE

    Args:
        override: Explicit override for verbose (True=always enable, False=always disable,
                  None=check environment variable, defaults to True if not set)

    Returns:
        True if verbose logging should be enabled, False otherwise.

    Example:
        # Disable verbose logging globally via environment variable
        export CREWAI_VERBOSE=false

        # Or in code
        flow = Flow(verbose=False)
        crew = Crew(verbose=False)
    """
    if override is not None:
        return override

    env_value = os.getenv("CREWAI_VERBOSE", "").lower()
    if env_value in ("false", "0"):
        return False
    if env_value in ("true", "1"):
        return True

    # Default to True if not set
    return True

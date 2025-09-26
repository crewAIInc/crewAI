"""Logging and warning utility functions for CrewAI."""

import contextlib
import io
import logging
import warnings
from collections.abc import Generator


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

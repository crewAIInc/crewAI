"""Structured JSON logging utilities for A2A module."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
from typing import Any


_log_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "log_context", default=None
)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs logs as JSON with consistent fields for log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        context = _log_context.get()
        if context is not None:
            log_data.update(context)

        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "context_id"):
            log_data["context_id"] = record.context_id
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "extension"):
            log_data["extension"] = record.extension
        if hasattr(record, "error"):
            log_data["error"] = record.error

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            ):
                continue
            if key not in log_data:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class LogContext:
    """Context manager for adding fields to all logs within a scope.

    Example:
        with LogContext(task_id="abc", context_id="xyz"):
            logger.info("Processing task")  # Includes task_id and context_id
    """

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        self._token: Any = None

    def __enter__(self) -> LogContext:
        current = _log_context.get() or {}
        new_context = {**current, **self._fields}
        self._token = _log_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        _log_context.reset(self._token)


def configure_json_logging(logger_name: str = "crewai.a2a") -> None:
    """Configure JSON logging for the A2A module.

    Args:
        logger_name: Logger name to configure.
    """
    logger = logging.getLogger(logger_name)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for structured JSON output.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

"""Flow execution context management.

This module provides context variables for tracking flow execution state across
async boundaries and nested function calls.
"""

import contextvars


current_flow_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flow_request_id", default=None
)

current_flow_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flow_id", default=None
)

current_flow_method_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "flow_method_name", default="unknown"
)

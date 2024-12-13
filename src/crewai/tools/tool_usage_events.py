from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class ToolUsageEvent(BaseModel):
    agent_key: str
    agent_role: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_class: str
    run_attempts: int | None = None
    delegations: int | None = None


class ToolUsageFinished(ToolUsageEvent):
    started_at: datetime
    finished_at: datetime
    from_cache: bool = False


class ToolUsageError(ToolUsageEvent):
    error: str

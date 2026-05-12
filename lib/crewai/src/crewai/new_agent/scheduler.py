"""Task scheduler — lets agents schedule one-time or recurring work.

Persists tasks to ``~/.crewai/scheduled_tasks.json`` and runs an asyncio
background loop that fires due tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_PERSIST_PATH = Path.home() / ".crewai" / "scheduled_tasks.json"

# ── Relative-time parser ────────────────────────────────────────

_RELATIVE_RE = re.compile(
    r"(?:in\s+)?(\d+)\s*(second|sec|minute|min|hour|hr|day)s?",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600,
    "day": 86400,
}


def parse_schedule_time(text: str) -> datetime | None:
    """Parse a human-friendly time string into a UTC datetime.

    Supports:
    - Relative: "in 5 minutes", "30 seconds", "2 hours"
    - ISO 8601: "2026-05-11T18:00:00Z"
    """
    text = text.strip()

    # Try relative first
    m = _RELATIVE_RE.search(text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        secs = amount * _UNIT_SECONDS.get(unit, 60)
        return datetime.now(timezone.utc) + timedelta(seconds=secs)

    # Try ISO
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    return None


# ── ScheduledTask model ─────────────────────────────────────────

class ScheduledTask(BaseModel):
    id: str = Field(default_factory=lambda: f"task-{uuid4().hex[:8]}")
    agent_name: str = ""
    description: str = ""
    schedule_type: str = "once"  # "once" or "recurring"
    next_run_at: str = ""       # ISO 8601 UTC
    interval_seconds: int | None = None  # for recurring
    status: str = "pending"     # pending, running, completed, failed, cancelled
    last_result: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── TaskScheduler ───────────────────────────────────────────────

class TaskScheduler:
    """Singleton scheduler that checks for due tasks every 30 seconds."""

    _instance: TaskScheduler | None = None

    def __new__(cls) -> TaskScheduler:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._tasks: list[ScheduledTask] = []
        self._callback: Callable[[ScheduledTask], Any] | None = None
        self._running = False
        self._bg_task: asyncio.Task[None] | None = None
        self._load()

    def set_callback(self, cb: Callable[[ScheduledTask], Any]) -> None:
        self._callback = cb

    # ── Persistence ──

    def _load(self) -> None:
        if _PERSIST_PATH.exists():
            try:
                data = json.loads(_PERSIST_PATH.read_text())
                self._tasks = [ScheduledTask(**t) for t in data]
            except Exception:
                self._tasks = []

    def _save(self) -> None:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            _PERSIST_PATH.write_text(
                json.dumps([t.model_dump() for t in self._tasks], indent=2)
            )
        except Exception as e:
            logger.warning(f"Failed to persist scheduled tasks: {e}")

    # ── CRUD ──

    def add(self, task: ScheduledTask) -> ScheduledTask:
        self._tasks.append(task)
        self._save()
        return task

    def cancel(self, task_id: str) -> bool:
        for t in self._tasks:
            if t.id == task_id and t.status == "pending":
                t.status = "cancelled"
                self._save()
                return True
        return False

    def list_tasks(self, include_done: bool = False) -> list[ScheduledTask]:
        if include_done:
            return list(self._tasks)
        return [t for t in self._tasks if t.status in ("pending", "running")]

    # ── Background loop ──

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            return
        self._running = True
        if loop is not None:
            self._bg_task = loop.create_task(self._loop())
        else:
            try:
                running_loop = asyncio.get_running_loop()
                self._bg_task = running_loop.create_task(self._loop())
            except RuntimeError:
                pass

    def stop(self) -> None:
        self._running = False
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(30)
                self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Scheduler tick error: {e}")

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for task in self._tasks:
            if task.status != "pending":
                continue
            try:
                due = datetime.fromisoformat(task.next_run_at)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if now >= due:
                task.status = "running"
                self._save()
                try:
                    if self._callback:
                        result = self._callback(task)
                        task.last_result = str(result) if result else "done"
                except Exception as e:
                    task.status = "failed"
                    task.last_result = str(e)
                    self._save()
                    continue

                if task.schedule_type == "recurring" and task.interval_seconds:
                    task.status = "pending"
                    task.next_run_at = (
                        now + timedelta(seconds=task.interval_seconds)
                    ).isoformat()
                else:
                    task.status = "completed"
                self._save()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — for testing only."""
        cls._instance = None


# ── ScheduleTaskTool ────────────────────────────────────────────

class ScheduleTaskArgs(BaseModel):
    description: str = Field(
        description="What the agent should do when the task fires"
    )
    when: str = Field(
        description=(
            "When to run. Accepts relative ('in 5 minutes', '2 hours') "
            "or ISO 8601 ('2026-05-11T18:00:00Z')"
        )
    )
    recurring_interval: str | None = Field(
        default=None,
        description=(
            "For recurring tasks, how often to repeat (e.g. '30 minutes', '1 hour'). "
            "Omit for one-time tasks."
        ),
    )


class ScheduleTaskTool(BaseTool):
    """Tool that lets an agent schedule future work."""

    name: str = "schedule_task"
    description: str = (
        "Schedule a task to be executed at a future time. "
        "Use this when you promise to do something later, "
        "need to set a reminder, or want to run recurring checks."
    )
    args_schema: type[BaseModel] = ScheduleTaskArgs
    agent_name: str = Field(default="", exclude=True)

    def _run(
        self,
        description: str,
        when: str,
        recurring_interval: str | None = None,
        **kwargs: Any,
    ) -> str:
        run_at = parse_schedule_time(when)
        if run_at is None:
            return (
                f"Could not parse time '{when}'. "
                "Use relative ('in 5 minutes') or ISO 8601 format."
            )

        schedule_type = "once"
        interval_seconds: int | None = None

        if recurring_interval:
            m = _RELATIVE_RE.search(recurring_interval)
            if m:
                amount = int(m.group(1))
                unit = m.group(2).lower()
                interval_seconds = amount * _UNIT_SECONDS.get(unit, 60)
                schedule_type = "recurring"

        task = ScheduledTask(
            agent_name=self.agent_name,
            description=description,
            schedule_type=schedule_type,
            next_run_at=run_at.isoformat(),
            interval_seconds=interval_seconds,
        )

        scheduler = TaskScheduler()
        scheduler.add(task)

        when_str = run_at.strftime("%Y-%m-%d %H:%M UTC")
        result = f"Scheduled task '{task.id}': {description} — due {when_str}"
        if schedule_type == "recurring":
            result += f" (repeats every {recurring_interval})"
        return result

"""Task information tracking for A2A integration."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import asyncio


@dataclass
class TaskInfo:
    """Information about a running task in the A2A executor.
    
    This class tracks the lifecycle and status of tasks being executed
    by the CrewAgentExecutor, providing better task management capabilities.
    
    Attributes:
        task: The asyncio task being executed
        started_at: When the task was started
        status: Current status of the task ("running", "completed", "cancelled", "failed")
    """
    task: asyncio.Task
    started_at: datetime
    status: str = "running"
    
    def update_status(self, new_status: str) -> None:
        """Update the task status.
        
        Args:
            new_status: The new status to set
        """
        self.status = new_status
    
    @property
    def is_running(self) -> bool:
        """Check if the task is currently running."""
        return self.status == "running" and not self.task.done()
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the task in seconds.
        
        Returns:
            Duration in seconds if task is completed, None if still running
        """
        if self.task.done():
            return (datetime.now() - self.started_at).total_seconds()
        return None

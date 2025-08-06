from datetime import datetime
from typing import Any
from crewai.crews.execution_trace import ExecutionStep, ExecutionTrace
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.agent_events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    AgentLogsExecutionEvent,
)
from crewai.utilities.events.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
)
from crewai.utilities.events.task_events import (
    TaskStartedEvent,
    TaskCompletedEvent,
)

class ExecutionTraceCollector:
    """Collects execution events and builds an execution trace."""
    
    def __init__(self):
        self.trace = ExecutionTrace()
        self.is_collecting = False
    
    def start_collecting(self) -> None:
        """Start collecting execution events."""
        self.is_collecting = True
        self.trace = ExecutionTrace(start_time=datetime.now())
        
        crewai_event_bus.register_handler(TaskStartedEvent, self._handle_task_started)
        crewai_event_bus.register_handler(TaskCompletedEvent, self._handle_task_completed)
        crewai_event_bus.register_handler(AgentExecutionStartedEvent, self._handle_agent_started)
        crewai_event_bus.register_handler(AgentExecutionCompletedEvent, self._handle_agent_completed)
        crewai_event_bus.register_handler(AgentLogsExecutionEvent, self._handle_agent_logs)
        crewai_event_bus.register_handler(ToolUsageStartedEvent, self._handle_tool_started)
        crewai_event_bus.register_handler(ToolUsageFinishedEvent, self._handle_tool_finished)
    
    def stop_collecting(self) -> ExecutionTrace:
        """Stop collecting and return the execution trace."""
        self.is_collecting = False
        self.trace.end_time = datetime.now()
        
        return self.trace
    
    
    def _handle_agent_started(self, source: Any, event: AgentExecutionStartedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="agent_execution_started",
            agent_role=event.agent.role if hasattr(event.agent, 'role') else None,
            task_description=getattr(event.task, 'description', None) if event.task else None,
            content={
                "task_prompt": event.task_prompt,
                "tools": [tool.name for tool in event.tools] if event.tools else [],
            }
        )
        self.trace.add_step(step)
    
    def _handle_agent_completed(self, source: Any, event: AgentExecutionCompletedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="agent_execution_completed",
            agent_role=event.agent.role if hasattr(event.agent, 'role') else None,
            content={
                "output": event.output,
            }
        )
        self.trace.add_step(step)
    
    def _handle_agent_logs(self, source: Any, event: AgentLogsExecutionEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="agent_thought",
            agent_role=event.agent_role,
            content={
                "formatted_answer": str(event.formatted_answer),
            }
        )
        self.trace.add_step(step)
    
    def _handle_tool_started(self, source: Any, event: ToolUsageStartedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="tool_call_started",
            agent_role=event.agent_role,
            content={
                "tool_name": event.tool_name,
                "tool_args": event.tool_args,
                "tool_class": event.tool_class,
            }
        )
        self.trace.add_step(step)
    
    def _handle_tool_finished(self, source: Any, event: ToolUsageFinishedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="tool_call_completed",
            agent_role=event.agent_role,
            content={
                "tool_name": event.tool_name,
                "output": event.output,
                "from_cache": event.from_cache,
                "duration": (event.finished_at - event.started_at).total_seconds() if hasattr(event, 'started_at') and hasattr(event, 'finished_at') else None,
            }
        )
        self.trace.add_step(step)
    
    def _handle_task_started(self, source: Any, event: TaskStartedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="task_started",
            task_description=getattr(event.task, 'description', None) if hasattr(event, 'task') and event.task else None,
            content={
                "task_id": getattr(event.task, 'id', None) if hasattr(event, 'task') and event.task else None,
                "context": getattr(event, 'context', None),
            }
        )
        self.trace.add_step(step)
    
    def _handle_task_completed(self, source: Any, event: TaskCompletedEvent) -> None:
        if not self.is_collecting:
            return
        
        step = ExecutionStep(
            timestamp=datetime.now(),
            step_type="task_completed",
            task_description=getattr(event.task, 'description', None) if hasattr(event, 'task') and event.task else None,
            content={
                "task_id": getattr(event.task, 'id', None) if hasattr(event, 'task') and event.task else None,
                "output": event.output.raw if hasattr(event, 'output') and event.output else None,
            }
        )
        self.trace.add_step(step)

from typing import Optional, Dict, Any
import logging

from crewai.utilities.events.crew_events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
)
from crewai.utilities.events.agent_events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
)
from crewai.utilities.events import (
    ToolUsageStartedEvent,
    ToolUsageErrorEvent,
)
from crewai.utilities.events.base_event_listener import BaseEventListener

try:
    import mlflow
    import mlflow.tracing
    MLFLOW_INSTALLED = True
except ImportError:
    MLFLOW_INSTALLED = False

logger = logging.getLogger(__name__)


class MLflowListener(BaseEventListener):
    """MLflow integration listener for CrewAI events"""
    
    def __init__(self):
        super().__init__()
        self._active_spans: Dict[str, Any] = {}
        self._autolog_enabled = False
    
    def setup_listeners(self, crewai_event_bus):
        if not MLFLOW_INSTALLED:
            logger.warning("MLflow not installed, skipping listener setup")
            return
        

            
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_kickoff_started(source, event: CrewKickoffStartedEvent):
            if not self._autolog_enabled:
                return
            try:
                span = mlflow.tracing.start_span(
                    name=f"Crew Execution: {event.crew_name or 'Unknown'}",
                    span_type="CHAIN"
                )
                span.set_inputs(event.inputs or {})
                self._active_spans[f"crew_{event.source_fingerprint or id(source)}"] = span
            except Exception as e:
                logger.warning(f"Failed to start MLflow span for crew: {e}")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_completed(source, event: CrewKickoffCompletedEvent):
            if not self._autolog_enabled:
                return
            try:
                span_key = f"crew_{event.source_fingerprint or id(source)}"
                if span_key in self._active_spans:
                    span = self._active_spans[span_key]
                    span.set_outputs({"result": str(event.output)})
                    span.set_status("OK")
                    span.end()
                    del self._active_spans[span_key]
            except Exception as e:
                logger.warning(f"Failed to end MLflow span for crew: {e}")

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_kickoff_failed(source, event: CrewKickoffFailedEvent):
            if not self._autolog_enabled:
                return
            try:
                span_key = f"crew_{event.source_fingerprint or id(source)}"
                if span_key in self._active_spans:
                    span = self._active_spans[span_key]
                    span.set_status("ERROR")
                    span.set_attribute("error", event.error)
                    span.end()
                    del self._active_spans[span_key]
            except Exception as e:
                logger.warning(f"Failed to end MLflow span for crew error: {e}")

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            if not self._autolog_enabled:
                return
            try:
                span = mlflow.tracing.start_span(
                    name=f"Agent: {event.agent.role}",
                    span_type="AGENT"
                )
                span.set_inputs({
                    "task": str(event.task),
                    "task_prompt": event.task_prompt,
                    "tools": [tool.name for tool in (event.tools or [])]
                })
                self._active_spans[f"agent_{event.source_fingerprint or id(event.agent)}"] = span
            except Exception as e:
                logger.warning(f"Failed to start MLflow span for agent: {e}")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            if not self._autolog_enabled:
                return
            try:
                span_key = f"agent_{event.source_fingerprint or id(event.agent)}"
                if span_key in self._active_spans:
                    span = self._active_spans[span_key]
                    span.set_outputs({"output": event.output})
                    span.set_status("OK")
                    span.end()
                    del self._active_spans[span_key]
            except Exception as e:
                logger.warning(f"Failed to end MLflow span for agent: {e}")

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def on_agent_execution_error(source, event: AgentExecutionErrorEvent):
            if not self._autolog_enabled:
                return
            try:
                span_key = f"agent_{event.source_fingerprint or id(event.agent)}"
                if span_key in self._active_spans:
                    span = self._active_spans[span_key]
                    span.set_status("ERROR")
                    span.set_attribute("error", event.error)
                    span.end()
                    del self._active_spans[span_key]
            except Exception as e:
                logger.warning(f"Failed to end MLflow span for agent error: {e}")

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            if not self._autolog_enabled:
                return
            try:
                span = mlflow.tracing.start_span(
                    name=f"Tool: {event.tool_name}",
                    span_type="TOOL"
                )
                span.set_inputs({"tool_name": event.tool_name})
                self._active_spans[f"tool_{id(event)}"] = span
            except Exception as e:
                logger.warning(f"Failed to start MLflow span for tool: {e}")

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            if not self._autolog_enabled:
                return
            try:
                span_key = f"tool_{id(event)}"
                if span_key in self._active_spans:
                    span = self._active_spans[span_key]
                    span.set_status("ERROR")
                    span.set_attribute("error", event.error)
                    span.end()
                    del self._active_spans[span_key]
            except Exception as e:
                logger.warning(f"Failed to end MLflow span for tool error: {e}")


mlflow_listener = MLflowListener()

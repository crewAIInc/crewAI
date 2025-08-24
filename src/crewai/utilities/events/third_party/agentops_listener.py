import logging

from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.crewai_event_bus import CrewAIEventsBus
from crewai.utilities.events.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.utilities.events.task_events import TaskEvaluationEvent
from crewai.utilities.events.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
)

logger = logging.getLogger(__name__)


class AgentOpsListener(BaseEventListener):
    def __init__(self):
        self.agentops = None
        try:
            import agentops

            self.agentops = agentops
            logger.info("AgentOps integration enabled")
        except ImportError:
            logger.debug("AgentOps not installed, skipping AgentOps integration")

        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus):
        if self.agentops is None:
            return

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_kickoff_started(source, event):
            self._handle_crew_kickoff_started(source, event)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_completed(source, event):
            self._handle_crew_kickoff_completed(source, event)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event):
            self._handle_tool_usage_started(source, event)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event):
            self._handle_tool_usage_error(source, event)

        @crewai_event_bus.on(TaskEvaluationEvent)
        def on_task_evaluation(source, event):
            self._handle_task_evaluation(source, event)

    def _handle_crew_kickoff_started(self, source, event: CrewKickoffStartedEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.start_session(
                tags=["crewai", "crew_kickoff"],
                config=self.agentops.Configuration(
                    auto_start_session=False,
                    instrument_llm_calls=True,
                ),
            )
            logger.debug("AgentOps session started for crew kickoff")
        except Exception as e:
            logger.warning(f"Failed to start AgentOps session: {e}")

    def _handle_crew_kickoff_completed(self, source, event: CrewKickoffCompletedEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.end_session("Success")
            logger.debug("AgentOps session ended for crew kickoff completion")
        except Exception as e:
            logger.warning(f"Failed to end AgentOps session: {e}")

    def _handle_tool_usage_started(self, source, event: ToolUsageStartedEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ActionEvent(
                    action_type="tool_usage",
                    params={
                        "tool_name": event.tool_name,
                        "tool_args": event.tool_args,
                    },
                )
            )
            logger.debug(f"AgentOps recorded tool usage: {event.tool_name}")
        except Exception as e:
            logger.warning(f"Failed to record tool usage in AgentOps: {e}")

    def _handle_tool_usage_error(self, source, event: ToolUsageErrorEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ErrorEvent(
                    message=f"Tool usage error: {event.error}",
                    error_type="ToolUsageError",
                    details={
                        "tool_name": event.tool_name,
                        "tool_args": event.tool_args,
                    },
                )
            )
            logger.debug(f"AgentOps recorded tool usage error: {event.tool_name}")
        except Exception as e:
            logger.warning(f"Failed to record tool usage error in AgentOps: {e}")

    def _handle_task_evaluation(self, source, event: TaskEvaluationEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ActionEvent(
                    action_type="task_evaluation",
                    params={
                        "evaluation_type": event.evaluation_type,
                        "task": str(event.task) if event.task else None,
                    },
                )
            )
            logger.debug(f"AgentOps recorded task evaluation: {event.evaluation_type}")
        except Exception as e:
            logger.warning(f"Failed to record task evaluation in AgentOps: {e}")


agentops_listener = AgentOpsListener()

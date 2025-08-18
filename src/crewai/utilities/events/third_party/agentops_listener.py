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

        crewai_event_bus.register_handler(
            CrewKickoffStartedEvent, self._handle_crew_kickoff_started
        )
        crewai_event_bus.register_handler(
            CrewKickoffCompletedEvent, self._handle_crew_kickoff_completed
        )
        crewai_event_bus.register_handler(
            ToolUsageStartedEvent, self._handle_tool_usage_started
        )
        crewai_event_bus.register_handler(
            ToolUsageErrorEvent, self._handle_tool_usage_error
        )
        crewai_event_bus.register_handler(
            TaskEvaluationEvent, self._handle_task_evaluation
        )

    def _handle_crew_kickoff_started(self, event: CrewKickoffStartedEvent):
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

    def _handle_crew_kickoff_completed(self, event: CrewKickoffCompletedEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.end_session("Success")
            logger.debug("AgentOps session ended for crew kickoff completion")
        except Exception as e:
            logger.warning(f"Failed to end AgentOps session: {e}")

    def _handle_tool_usage_started(self, event: ToolUsageStartedEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ActionEvent(
                    action_type="tool_usage",
                    params={
                        "tool_name": event.tool_name,
                        "arguments": event.arguments,
                    },
                )
            )
            logger.debug(f"AgentOps recorded tool usage: {event.tool_name}")
        except Exception as e:
            logger.warning(f"Failed to record tool usage in AgentOps: {e}")

    def _handle_tool_usage_error(self, event: ToolUsageErrorEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ErrorEvent(
                    message=f"Tool usage error: {event.error}",
                    error_type="ToolUsageError",
                    details={
                        "tool_name": event.tool_name,
                        "arguments": event.arguments,
                    },
                )
            )
            logger.debug(f"AgentOps recorded tool usage error: {event.tool_name}")
        except Exception as e:
            logger.warning(f"Failed to record tool usage error in AgentOps: {e}")

    def _handle_task_evaluation(self, event: TaskEvaluationEvent):
        if self.agentops is None:
            return

        try:
            self.agentops.record(
                self.agentops.ActionEvent(
                    action_type="task_evaluation",
                    params={
                        "task_id": str(event.task_id),
                        "score": event.score,
                        "feedback": event.feedback,
                    },
                )
            )
            logger.debug(f"AgentOps recorded task evaluation: {event.task_id}")
        except Exception as e:
            logger.warning(f"Failed to record task evaluation in AgentOps: {e}")


agentops_listener = AgentOpsListener()

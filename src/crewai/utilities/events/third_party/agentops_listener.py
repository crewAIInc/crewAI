from typing import Optional

from crewai.utilities.events import (
    CrewKickoffCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.crew_events import CrewKickoffStartedEvent
from crewai.utilities.events.task_events import TaskEvaluationEvent

try:
    import agentops

    AGENTOPS_INSTALLED = True
except ImportError:
    AGENTOPS_INSTALLED = False


class AgentOpsListener(BaseEventListener):
    tool_event: Optional["agentops.ToolEvent"] = None
    session: Optional["agentops.Session"] = None

    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        if not AGENTOPS_INSTALLED:
            return

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_kickoff_started(source, event: CrewKickoffStartedEvent):
            self.session = agentops.init()
            for agent in source.agents:
                if self.session:
                    self.session.create_agent(
                        name=agent.role,
                        agent_id=str(agent.id),
                    )

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_completed(source, event: CrewKickoffCompletedEvent):
            if self.session:
                self.session.end_session(
                    end_state="Success",
                    end_state_reason="Finished Execution",
                )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            self.tool_event = agentops.ToolEvent(name=event.tool_name)
            if self.session:
                self.session.record(self.tool_event)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            agentops.ErrorEvent(exception=event.error, trigger_event=self.tool_event)

        @crewai_event_bus.on(TaskEvaluationEvent)
        def on_task_evaluation(source, event: TaskEvaluationEvent):
            if self.session:
                self.session.create_agent(
                    name="Task Evaluator", agent_id=str(source.original_agent.id)
                )


agentops_listener = AgentOpsListener()

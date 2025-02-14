from typing import Optional
from crewai.utilities.events.agent_events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent
)
from crewai.utilities.events.base_event_listener import BaseEventListener

from crewai.utilities.events import ToolUsageErrorEvent, ToolUsageStartedEvent, CrewKickoffCompletedEvent
from crewai.utilities.events.task_events import TaskEvaluationEvent

try:
    import agentops
    from agentops import Client
    AGENTOPS_INSTALLED = True
except ImportError:
    AGENTOPS_INSTALLED = False
    def track_agent():
        def noop(f):
            return f
        return noop

class AgentOpsListener(BaseEventListener):
    tool_event: Optional[agentops.ToolEvent] = None
    
    def __init__(self):
        super().__init__()
        if AGENTOPS_INSTALLED:
            self.session =agentops.init()

    def setup_listeners(self, event_bus):
      if not AGENTOPS_INSTALLED:
        return 
      
      @event_bus.on(AgentExecutionStartedEvent)
      def on_agent_started(source, event: AgentExecutionStartedEvent):
              agent = event.agent
              Client().create_agent(name=agent.role, agent_id=str(agent.id))

      @event_bus.on(CrewKickoffCompletedEvent)
      def on_agent_error(source, event: CrewKickoffCompletedEvent):
            agentops.end_session(
                end_state="Success",
                end_state_reason="Finished Execution",
                is_auto_end=True
            )
            
      @event_bus.on(ToolUsageStartedEvent)
      def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            self.tool_event = agentops.ToolEvent(name=event.tool_name) 
            
      @event_bus.on(ToolUsageErrorEvent)
      def on_tool_usage_error(source, event: ToolUsageErrorEvent):
          agentops.ErrorEvent(exception=event.error, trigger_event=self.tool_event)
          
      @event_bus.on(TaskEvaluationEvent)
      def on_task_evaluation(source, event: TaskEvaluationEvent):
          Client().create_agent(name="Task Evaluator", agent_id=str(source.original_agent.id))

agentops_listener = AgentOpsListener()

from crewai.utilities.events.agent_events import AgentExecutionStarted
from crewai.utilities.events.base_event_listener import BaseEventListener

try:
    import agentops

    AGENTOPS_INSTALLED = True
except ImportError:
    AGENTOPS_INSTALLED = False


class AgentOpsListener(BaseEventListener):
    def __init__(self):
        super().__init__()
        print("AgentOpsListener init")

    def setup_listeners(self, event_bus):
        if AGENTOPS_INSTALLED:

            @event_bus.on(AgentExecutionStarted)
            def on_agent_started(source, event: AgentExecutionStarted):
                print("AGENTOPS WORKS !!!", event.agent)


agentops_listener = AgentOpsListener()

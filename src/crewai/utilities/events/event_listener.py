from pydantic import PrivateAttr

from crewai.telemetry.telemetry import Telemetry
from crewai.utilities import Logger
from crewai.utilities.constants import EMITTER_COLOR
from crewai.utilities.events.base_event_listener import BaseEventListener

from .agent_events import AgentExecutionCompletedEvent, AgentExecutionStartedEvent
from .crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestFailedEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainFailedEvent,
    CrewTrainStartedEvent,
)
from .flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from .task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent
from .tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


class EventListener(BaseEventListener):
    _instance = None
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())
    logger = Logger(verbose=True, default_color=EMITTER_COLOR)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized") or not self._initialized:
            super().__init__()
            self._telemetry = Telemetry()
            self._telemetry.set_tracer()
            self._initialized = True

    # Crew Events: kickoff, test, train
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            self.logger.log(
                f"🚀 Crew '{event.crew_name}' started",
                event.timestamp,
            )
            self._telemetry.crew_execution_span(source, event.inputs)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event: CrewKickoffFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTestStartedEvent)
        def on_crew_test_started(source, event: CrewTestStartedEvent):
            cloned_crew = source.copy()
            cloned_crew._telemetry.test_execution_span(
                cloned_crew,
                event.n_iterations,
                event.inputs,
                event.openai_model_name,
            )
            self.logger.log(
                f"🚀 Crew '{event.crew_name}' started test",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTestCompletedEvent)
        def on_crew_test_completed(source, event: CrewTestCompletedEvent):
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed test",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTestFailedEvent)
        def on_crew_test_failed(source, event: CrewTestFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed test",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTrainStartedEvent)
        def on_crew_train_started(source, event: CrewTrainStartedEvent):
            self.logger.log(
                f"📋 Crew '{event.crew_name}' started train",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTrainCompletedEvent)
        def on_crew_train_completed(source, event: CrewTrainCompletedEvent):
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed train",
                event.timestamp,
            )

        @crewai_event_bus.on(CrewTrainFailedEvent)
        def on_crew_train_failed(source, event: CrewTrainFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed train",
                event.timestamp,
            )

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            source._execution_span = self._telemetry.task_started(
                crew=source.agent.crew, task=source
            )
            self.logger.log(
                f"📋 Task started: {source.description}",
                event.timestamp,
            )

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event):
            if source._execution_span:
                self._telemetry.task_ended(
                    source._execution_span, source, source.agent.crew
                )
            self.logger.log(
                f"✅ Task completed: {source.description}",
                event.timestamp,
            )
            source._execution_span = None

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event: TaskFailedEvent):
            if source._execution_span:
                if source.agent and source.agent.crew:
                    self._telemetry.task_ended(
                        source._execution_span, source, source.agent.crew
                    )
                source._execution_span = None
            self.logger.log(
                f"❌ Task failed: {source.description}",
                event.timestamp,
            )

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            self.logger.log(
                f"🤖 Agent '{event.agent.role}' started task",
                event.timestamp,
            )

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            self.logger.log(
                f"✅ Agent '{event.agent.role}' completed task",
                event.timestamp,
            )

        # Flow Events

        @crewai_event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event: FlowCreatedEvent):
            self._telemetry.flow_creation_span(self.__class__.__name__)
            self.logger.log(
                f"🌊 Flow Created: '{event.flow_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event: FlowStartedEvent):
            self._telemetry.flow_execution_span(
                source.__class__.__name__, list(source._methods.keys())
            )
            self.logger.log(
                f"🤖 Flow Started: '{event.flow_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event: FlowFinishedEvent):
            self.logger.log(
                f"👍 Flow Finished: '{event.flow_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event: MethodExecutionStartedEvent):
            self.logger.log(
                f"🤖 Flow Method Started: '{event.method_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def on_method_execution_failed(source, event: MethodExecutionFailedEvent):
            self.logger.log(
                f"❌ Flow Method Failed: '{event.method_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event: MethodExecutionFinishedEvent):
            self.logger.log(
                f"👍 Flow Method Finished: '{event.method_name}'",
                event.timestamp,
            )

        # Tool Usage Events
        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            self.logger.log(
                f"🤖 Tool Usage Started: '{event.tool_name}'",
                event.timestamp,
            )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            self.logger.log(
                f"✅ Tool Usage Finished: '{event.tool_name}'",
                event.timestamp,
                #
            )

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            self.logger.log(
                f"❌ Tool Usage Error: '{event.tool_name}'",
                event.timestamp,
                #
            )


event_listener = EventListener()

from pydantic import PrivateAttr

from crewai.telemetry.telemetry import Telemetry
from crewai.utilities import Logger
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
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from .task_events import TaskCompletedEvent, TaskFailedEvent, TaskStartedEvent


class EventListener(BaseEventListener):
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())
    logger = Logger(verbose=True)
    color = "bold_blue"

    def __init__(self):
        super().__init__()
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()

    # Crew Events: kickoff, test, train
    def setup_listeners(self, event_bus):
        @event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            self.logger.log(
                f"🚀 Crew '{event.crew_name}' started",
                event.timestamp,
                color=self.color,
            )
            self._telemetry.crew_execution_span(source, event.inputs)

        @event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event: CrewKickoffFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewTestStartedEvent)
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
                color=self.color,
            )

        @event_bus.on(CrewTestCompletedEvent)
        def on_crew_test_completed(source, event: CrewTestCompletedEvent):
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed test",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewTestFailedEvent)
        def on_crew_test_failed(source, event: CrewTestFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed test",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewTrainStartedEvent)
        def on_crew_train_started(source, event: CrewTrainStartedEvent):
            self.logger.log(
                f"📋 Crew '{event.crew_name}' started train",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewTrainCompletedEvent)
        def on_crew_train_completed(source, event: CrewTrainCompletedEvent):
            self.logger.log(
                f"✅ Crew '{event.crew_name}' completed train",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(CrewTrainFailedEvent)
        def on_crew_train_failed(source, event: CrewTrainFailedEvent):
            self.logger.log(
                f"❌ Crew '{event.crew_name}' failed train",
                event.timestamp,
                color=self.color,
            )
        
        @event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            source._execution_span = self._telemetry.task_started(
                crew=source.agent.crew, task=source
            )
            context = event.context
            self.logger.log(
                f"📋 Task started: {source.description} Context: {context}",
                event.timestamp,
                color=self.color,
            )
            
         
        
        @event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event: TaskCompletedEvent):
            if source._execution_span:
                self._telemetry.task_ended(source._execution_span, source, source.agent.crew)
            self.logger.log(
                f"📋 Task completed: {source.description}",
                event.timestamp,
                color=self.color,
            )
            source._execution_span = None
            
        @event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event: TaskFailedEvent):
            if source._execution_span:
                if source.agent and source.agent.crew:
                    self._telemetry.task_ended(source._execution_span, source, source.agent.crew)
                source._execution_span = None
            self.logger.log(
                f"❌ Task failed: {source.description}",
                event.timestamp,
                color=self.color,
            )


        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            self.logger.log(
                f"🤖 Agent '{event.agent.role}' started task",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            self.logger.log(
                f"👍 Agent '{event.agent.role}' completed task",
                event.timestamp,
                color=self.color,
            )

        # Flow Events
        
        @event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event: FlowCreatedEvent):
            self._telemetry.flow_creation_span(self.__class__.__name__)
            self.logger.log(
                f"🌊 Flow Created: '{event.flow_name}'",
                event.timestamp,
                color=self.color,
            )
        
        @event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event: FlowStartedEvent):
            self._telemetry.flow_execution_span(
                source.__class__.__name__, list(source._methods.keys())
            )
            self.logger.log(
                f"🤖 Flow Started: '{event.flow_name}'",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event: FlowFinishedEvent):
            self.logger.log(
                f"👍 Flow Finished: '{event.flow_name}'",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event: MethodExecutionStartedEvent):
            self.logger.log(
                f"🤖 Flow Method Started: '{event.method_name}'",
                event.timestamp,
                color=self.color,
            )

        @event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event: MethodExecutionFinishedEvent):
            self.logger.log(
                f"👍 Flow Method Finished: '{event.method_name}'",
                event.timestamp,
                color=self.color,
            )


event_listener = EventListener()

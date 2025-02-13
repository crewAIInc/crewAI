from pydantic import PrivateAttr

from crewai.telemetry.telemetry import Telemetry
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.events.base_event_listener import BaseEventListener

from .agent_events import AgentExecutionCompletedEvent, AgentExecutionStartedEvent
from .crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
    CrewTestCompletedEvent,
    CrewTestStartedEvent,
)
from .flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from .task_events import TaskCompletedEvent, TaskStartedEvent


class EventListener(BaseEventListener):
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())

    def __init__(self):
        super().__init__()
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()

    def setup_listeners(self, event_bus):
        @event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            print(f"🚀 Crew '{event.crew_name}' started", event.timestamp)
            print("event.inputs", event.inputs)
            self._telemetry.crew_execution_span(source, event.inputs)

        @event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)

        @event_bus.on(CrewTestStartedEvent)
        def on_crew_test_started(source, event: CrewTestStartedEvent):
            cloned_crew = source.copy()
            cloned_crew._telemetry.test_execution_span(
                cloned_crew,
                event.n_iterations,
                event.inputs,
                event.openai_model_name,
            )
            print(f"🚀 Crew '{event.crew_name}' started test")

        @event_bus.on(CrewTestCompletedEvent)
        def on_crew_test_completed(source, event: CrewTestCompletedEvent):
            print(f"👍 Crew '{event.crew_name}' completed test")

        @event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            print(f"📋 Task started: {event.task.description}")

        @event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event: TaskCompletedEvent):
            print(f"   Output: {event.output}")
            result = TaskEvaluator(event.task.agent).evaluate(event.task, event.output)
            print(f"   Evaluation: {result.quality}")
            if result.quality > 5:
                print(f" ✅ Passed: {result.suggestions}")
            else:
                print(f" ❌ Failed: {result.suggestions}")

        @event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            print(f"🤖 Agent '{event.agent.role}' started task")

        @event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            print(f"👍 Agent '{event.agent.role}' completed task")
            print(f"   Output: {event.output}")

        @event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event: FlowStartedEvent):
            print(f"🤖 Flow Started: '{event.flow_name}'")

        @event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event: FlowFinishedEvent):
            print(f"👍 Flow Finished: '{event.flow_name}'")

        @event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event: MethodExecutionStartedEvent):
            print(f"🤖 Flow Method Started: '{event.method_name}'")

        @event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event: MethodExecutionFinishedEvent):
            print(f"👍 Flow Method Finished: '{event.method_name}'")


event_listener = EventListener()

from pydantic import PrivateAttr

from crewai.telemetry.telemetry import Telemetry
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.events.base_event_listener import BaseEventListener

from .agent_events import (
    AgentExecutionCompleted,
    AgentExecutionStarted,
)
from .crew_events import (
    CrewKickoffCompleted,
    CrewKickoffStarted,
    CrewTestCompleted,
    CrewTestStarted,
)
from .flow_events import (
    FlowFinished,
    FlowStarted,
    MethodExecutionFinished,
    MethodExecutionStarted,
)
from .task_events import TaskCompleted, TaskStarted


class EventListener(BaseEventListener):
    _telemetry: Telemetry = PrivateAttr(default_factory=lambda: Telemetry())

    def __init__(self):
        super().__init__()
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()

    def setup_listeners(self, event_bus):
        @event_bus.on(CrewKickoffStarted)
        def on_crew_started(source, event: CrewKickoffStarted):
            print(f"🚀 Crew '{event.crew_name}' started", event.timestamp)
            print("event.inputs", event.inputs)
            self._telemetry.crew_execution_span(source, event.inputs)

        @event_bus.on(CrewKickoffCompleted)
        def on_crew_completed(source, event: CrewKickoffCompleted):
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)

        @event_bus.on(CrewTestStarted)
        def on_crew_test_started(source, event: CrewTestStarted):
            cloned_crew = source.copy()
            cloned_crew._telemetry.test_execution_span(
                cloned_crew,
                event.n_iterations,
                event.inputs,
                event.openai_model_name,
            )
            print(f"🚀 Crew '{event.crew_name}' started test")

        @event_bus.on(CrewTestCompleted)
        def on_crew_test_completed(source, event: CrewTestCompleted):
            print(f"👍 Crew '{event.crew_name}' completed test")

        @event_bus.on(TaskStarted)
        def on_task_started(source, event: TaskStarted):
            print(f"📋 Task started: {event.task.description}")

        @event_bus.on(TaskCompleted)
        def on_task_completed(source, event: TaskCompleted):
            print(f"   Output: {event.output}")
            result = TaskEvaluator(event.task.agent).evaluate(event.task, event.output)
            print(f"   Evaluation: {result.quality}")
            if result.quality > 5:
                print(f" ✅ Passed: {result.suggestions}")
            else:
                print(f" ❌ Failed: {result.suggestions}")

        @event_bus.on(AgentExecutionStarted)
        def on_agent_execution_started(source, event: AgentExecutionStarted):
            print(f"🤖 Agent '{event.agent.role}' started task")

        @event_bus.on(AgentExecutionCompleted)
        def on_agent_execution_completed(source, event: AgentExecutionCompleted):
            print(f"👍 Agent '{event.agent.role}' completed task")
            print(f"   Output: {event.output}")

        @event_bus.on(FlowStarted)
        def on_flow_started(source, event: FlowStarted):
            print(f"🤖 Flow Started: '{event.flow_name}'")

        @event_bus.on(FlowFinished)
        def on_flow_finished(source, event: FlowFinished):
            print(f"👍 Flow Finished: '{event.flow_name}'")

        @event_bus.on(MethodExecutionStarted)
        def on_method_execution_started(source, event: MethodExecutionStarted):
            print(f"🤖 Flow Method Started: '{event.method_name}'")

        @event_bus.on(MethodExecutionFinished)
        def on_method_execution_finished(source, event: MethodExecutionFinished):
            print(f"👍 Flow Method Finished: '{event.method_name}'")


event_listener = EventListener()

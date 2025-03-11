from datetime import datetime
from io import StringIO
from typing import Any, Dict, Optional

from pydantic import Field, PrivateAttr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from crewai.task import Task
from crewai.telemetry.telemetry import Telemetry
from crewai.utilities import Logger
from crewai.utilities.constants import EMITTER_COLOR
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMStreamChunkEvent,
)

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
    execution_spans: Dict[Task, Any] = Field(default_factory=dict)
    next_chunk = 0
    text_stream = StringIO()
    current_crew_tree = None
    current_task_branch = None
    current_agent_branch = None
    current_tool_branch = None
    current_flow_tree = None
    current_method_branch = None
    tool_usage_counts: Dict[str, int] = {}

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
            self.execution_spans = {}
            self._initialized = True
            self.console = Console(width=None)
            self.tool_usage_counts = {}

    def _format_timestamp(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    def _create_panel(self, content: Text, title: str, style: str = "blue") -> Panel:
        """Create a standardized panel with consistent styling."""
        return Panel(
            content,
            title=title,
            border_style=style,
            padding=(1, 2),
        )

    def _create_status_content(
        self, title: str, name: str, status_style: str = "blue", **fields
    ) -> Text:
        """Create standardized status content with consistent formatting."""
        content = Text()
        content.append(f"{title}\n", style=f"{status_style} bold")
        content.append("Name: ", style="white")
        content.append(f"{name}\n", style=status_style)

        for label, value in fields.items():
            content.append(f"{label}: ", style="white")
            content.append(
                f"{value}\n", style=fields.get(f"{label}_style", status_style)
            )

        return content

    def _update_tree_label(
        self,
        tree: Tree,
        prefix: str,
        name: str,
        style: str = "blue",
        status: Optional[str] = None,
    ) -> None:
        """Update tree label with consistent formatting."""
        label = Text()
        label.append(f"{prefix} ", style=f"{style} bold")
        label.append(name, style=style)
        if status:
            label.append("\n    Status: ", style="white")
            label.append(status, style=f"{style} bold")
        tree.label = label

    def _add_tree_node(self, parent: Tree, text: str, style: str = "yellow") -> Tree:
        """Add a node to the tree with consistent styling."""
        return parent.add(Text(text, style=style))

    # ----------- METHODS -----------

    def on_crew_start(self, source: Any, event: Any) -> None:
        if self.verbose:
            self.current_crew_tree = Tree(
                Text("🚀 Crew: ", style="cyan bold")
                + Text(event.crew_name, style="cyan")
            )

            content = self._create_status_content(
                "Crew Execution Started",
                event.crew_name,
                "cyan",
                ID=source.id,
            )

            panel = self._create_panel(content, "Crew Execution Started", "cyan")
            self.console.print(panel)
            self.console.print()

        self._telemetry.crew_execution_span(source, event.inputs)

    # ----------- CREW EVENTS -----------

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event: CrewKickoffStartedEvent):
            self.on_crew_start(source, event)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event: CrewKickoffCompletedEvent):
            # Handle telemetry
            final_string_output = event.output.raw
            self._telemetry.end_crew(source, final_string_output)
            if self.verbose:
                if self.current_crew_tree:
                    self._update_tree_label(
                        self.current_crew_tree,
                        "✅ Crew:",
                        event.crew_name or "Crew",
                        "green",
                    )

                    completion_content = self._create_status_content(
                        "Crew Execution Completed",
                        event.crew_name or "Crew",
                        "green",
                        ID=source.id,
                    )

                    self.console.print(self.current_crew_tree)
                    self.console.print()
                    panel = self._create_panel(
                        completion_content, "Crew Completion", "green"
                    )
                    self.console.print(panel)
                    self.console.print()

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event: CrewKickoffFailedEvent):
            if self.verbose:
                if self.current_crew_tree:
                    # Update crew tree label to show failure
                    crew_content = Text()
                    crew_content.append("❌ Crew: ", style="red bold")
                    crew_content.append(event.crew_name or "Crew", style="red")
                    self.current_crew_tree.label = crew_content

                    # Create failure panel
                    failure_content = Text()
                    failure_content.append("Crew Execution Failed\n", style="red bold")
                    failure_content.append("Name: ", style="white")
                    failure_content.append(f"{event.crew_name}\n", style="red")
                    failure_content.append("ID: ", style="white")
                    failure_content.append(str(source.id), style="blue")

                    # Show final tree and failure panel
                    self.console.print(self.current_crew_tree)
                    self.console.print()

                    panel = self._create_panel(failure_content, "Crew Failure", "red")
                    self.console.print(panel)
                    self.console.print()

        @crewai_event_bus.on(CrewTestFailedEvent)
        def on_crew_test_failed(source, event: CrewTestFailedEvent):
            if self.verbose:
                failure_content = Text()
                failure_content.append("❌ Crew Test Failed\n", style="red bold")
                failure_content.append("Crew: ", style="white")
                failure_content.append(event.crew_name or "Crew", style="red")

                panel = self._create_panel(failure_content, "Test Failure", "red")
                self.console.print(panel)
                self.console.print()

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
            if self.verbose:
                failure_content = Text()
                failure_content.append("❌ Crew Training Failed\n", style="red bold")
                failure_content.append("Crew: ", style="white")
                failure_content.append(event.crew_name or "Crew", style="red")

                panel = self._create_panel(failure_content, "Training Failure", "red")
                self.console.print(panel)
                self.console.print()

        # ----------- TASK EVENTS -----------

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source, event: TaskStartedEvent):
            span = self._telemetry.task_started(crew=source.agent.crew, task=source)
            self.execution_spans[source] = span

            if self.verbose:
                task_content = Text()
                task_content.append(f"📋 Task: {source.id}", style="yellow bold")
                task_content.append("\n   Status: ", style="white")
                task_content.append("Executing Task...", style="yellow dim")

                # Add task to the crew tree
                if self.current_crew_tree:
                    self.current_task_branch = self.current_crew_tree.add(task_content)
                    self.console.print(self.current_crew_tree)
                else:
                    panel = self._create_panel(task_content, "Task Started", "yellow")
                    self.console.print(panel)

                self.console.print()

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source, event: TaskCompletedEvent):
            # Handle telemetry
            span = self.execution_spans.get(source)
            if span:
                self._telemetry.task_ended(span, source, source.agent.crew)
            self.execution_spans[source] = None

            if self.verbose:
                if self.current_crew_tree:
                    for branch in self.current_crew_tree.children:
                        if str(source.id) in str(branch.label):
                            task_content = Text()
                            task_content.append(
                                f"📋 Task: {source.id}", style="green bold"
                            )
                            task_content.append("\n   Assigned to: ", style="white")
                            task_content.append(source.agent.role, style="green")
                            task_content.append("\n   Status: ", style="white")
                            task_content.append("✅ Completed", style="green bold")
                            branch.label = task_content
                            self.console.print(self.current_crew_tree)
                            break

                completion_content = self._create_status_content(
                    "Task Completed", str(source.id), "green", Agent=source.agent.role
                )

                panel = self._create_panel(
                    completion_content, "Task Completion", "green"
                )
                self.console.print(panel)
                self.console.print()

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event: TaskFailedEvent):
            span = self.execution_spans.get(source)
            if span:
                if source.agent and source.agent.crew:
                    self._telemetry.task_ended(span, source, source.agent.crew)
                self.execution_spans[source] = None

            if self.verbose:
                failure_content = Text()
                failure_content.append("❌ Task Failed\n", style="red bold")
                failure_content.append(f"Task: {source.id}", style="white")
                failure_content.append(source.description, style="red")
                if source.agent:
                    failure_content.append("\nAgent: ", style="white")
                    failure_content.append(source.agent.role, style="red")

                # Update the tree if it exists
                if self.current_crew_tree:
                    # Find the task branch and update it with failure status
                    for branch in self.current_crew_tree.children:
                        if source.description in branch.label:
                            branch.label = Text("❌ ", style="red bold") + branch.label
                            self.console.print(self.current_crew_tree)
                            break

                # Show failure panel
                panel = self._create_panel(failure_content, "Task Failure", "red")
                self.console.print(panel)
                self.console.print()

        # ----------- AGENT EVENTS -----------

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def on_agent_execution_started(source, event: AgentExecutionStartedEvent):
            if self.verbose:
                if self.current_task_branch:
                    # Create agent execution branch with empty label
                    self.current_agent_branch = self.current_task_branch.add("")
                    self._update_tree_label(
                        self.current_agent_branch,
                        "🤖 Agent:",
                        event.agent.role,
                        "green",
                        "In Progress",
                    )

                    self.console.print(self.current_crew_tree)
                    self.console.print()

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event: AgentExecutionCompletedEvent):
            if self.verbose:
                if self.current_agent_branch:
                    self._update_tree_label(
                        self.current_agent_branch,
                        "🤖 Agent:",
                        event.agent.role,
                        "green",
                        "✅ Completed",
                    )

                    self.console.print(self.current_crew_tree)
                    self.console.print()

        # ----------- FLOW EVENTS -----------

        @crewai_event_bus.on(FlowCreatedEvent)
        def on_flow_created(source, event: FlowCreatedEvent):
            self._telemetry.flow_creation_span(event.flow_name)

            content = self._create_status_content(
                "Starting Flow Execution", event.flow_name, "blue"
            )

            panel = self._create_panel(content, "Flow Execution", "blue")

            self.console.print()
            self.console.print(panel)
            self.console.print()

            # Create and display the initial tree
            flow_label = Text()
            flow_label.append("🌊 Flow: ", style="blue bold")
            flow_label.append(event.flow_name, style="blue")

            self.current_flow_tree = Tree(flow_label)

            # Add both creation steps to show progression
            self.current_flow_tree.add(Text("✨ Created", style="blue"))
            self.current_flow_tree.add(
                Text("✅ Initialization Complete", style="green")
            )

            self.console.print(self.current_flow_tree)
            self.console.print()

        @crewai_event_bus.on(FlowStartedEvent)
        def on_flow_started(source, event: FlowStartedEvent):
            self._telemetry.flow_execution_span(
                event.flow_name, list(source._methods.keys())
            )
            self.current_flow_tree = Tree("")
            self._update_tree_label(
                self.current_flow_tree,
                "🌊 Flow:",
                event.flow_name,
                "blue",
                "In Progress",
            )

            # Add initial thinking state
            self.current_flow_tree.add(Text("🧠 Initializing...", style="yellow"))

            self.console.print()
            self.console.print(self.current_flow_tree)
            self.console.print()

        @crewai_event_bus.on(FlowFinishedEvent)
        def on_flow_finished(source, event: FlowFinishedEvent):
            if self.current_flow_tree:
                self._update_tree_label(
                    self.current_flow_tree,
                    "✅ Flow Finished:",
                    event.flow_name,
                    "green",
                )

                content = self._create_status_content(
                    "Flow Execution Completed",
                    event.flow_name,
                    "green",
                    ID=source.flow_id,
                )

                panel = self._create_panel(content, "Flow Completion", "green")
                self.console.print(panel)
                self.console.print()

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_method_execution_started(source, event: MethodExecutionStartedEvent):
            if self.current_flow_tree:
                # Find and update the method branch
                for branch in self.current_flow_tree.children:
                    if event.method_name in branch.label:
                        self.current_method_branch = branch
                        branch.label = Text("🔄 Running: ", style="yellow bold") + Text(
                            event.method_name, style="yellow"
                        )
                        break

                self.console.print(self.current_flow_tree)
                self.console.print()

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_method_execution_finished(source, event: MethodExecutionFinishedEvent):
            if self.current_method_branch:
                # Update method status
                self.current_method_branch.label = Text(
                    "✅ Completed: ", style="green bold"
                ) + Text(event.method_name, style="green")
                self.console.print(self.current_flow_tree)
                self.console.print()

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def on_method_execution_failed(source, event: MethodExecutionFailedEvent):
            if self.current_method_branch:
                self.current_method_branch.label = Text(
                    "❌ Failed: ", style="red bold"
                ) + Text(event.method_name, style="red")
                self.console.print(self.current_flow_tree)
                self.console.print()

        # ----------- TOOL USAGE EVENTS -----------

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_usage_started(source, event: ToolUsageStartedEvent):
            if self.verbose and self.current_agent_branch:
                # Update tool usage count
                self.tool_usage_counts[event.tool_name] = (
                    self.tool_usage_counts.get(event.tool_name, 0) + 1
                )

                # Find existing tool node or create new one
                tool_node = None
                for child in self.current_agent_branch.children:
                    if event.tool_name in child.label.plain:
                        tool_node = child
                        break

                if not tool_node:
                    # Create new tool node
                    self.current_tool_branch = self.current_agent_branch.add("")
                else:
                    self.current_tool_branch = tool_node

                # Update label with current count
                self._update_tree_label(
                    self.current_tool_branch,
                    "🔧",
                    f"Using {event.tool_name} ({self.tool_usage_counts[event.tool_name]})",
                    "yellow",
                )

                self.console.print(self.current_crew_tree)
                self.console.print()

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_usage_finished(source, event: ToolUsageFinishedEvent):
            if self.verbose and self.current_tool_branch:
                self._update_tree_label(
                    self.current_tool_branch,
                    "🔧",
                    f"Used {event.tool_name} ({self.tool_usage_counts[event.tool_name]})",
                    "green",
                )
                self.console.print(self.current_crew_tree)
                self.console.print()

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolUsageErrorEvent):
            if self.verbose:
                if self.current_tool_branch:
                    self._update_tree_label(
                        self.current_tool_branch,
                        "🔧 Failed",
                        f"{event.tool_name} ({self.tool_usage_counts[event.tool_name]})",
                        "red",
                    )
                    self.console.print(self.current_crew_tree)
                    self.console.print()

                # Show error panel
                error_content = self._create_status_content(
                    "Tool Usage Failed", event.tool_name, "red", Error=event.error
                )
                panel = self._create_panel(error_content, "Tool Error", "red")
                self.console.print(panel)
                self.console.print()

        # ----------- LLM EVENTS -----------

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_call_started(source, event: LLMCallStartedEvent):
            if self.verbose and self.current_agent_branch:
                if not any(
                    "Thinking" in str(child.label)
                    for child in self.current_agent_branch.children
                ):
                    self.current_tool_branch = self.current_agent_branch.add("")
                    self._update_tree_label(
                        self.current_tool_branch, "🧠", "Thinking...", "blue"
                    )
                self.console.print(self.current_crew_tree)
                self.console.print()

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call_completed(source, event: LLMCallCompletedEvent):
            if self.verbose and self.current_tool_branch:
                # Remove the thinking status node when complete
                if "Thinking" in str(self.current_tool_branch.label):
                    if self.current_agent_branch:
                        self.current_agent_branch.children.remove(
                            self.current_tool_branch
                        )
                    self.console.print(self.current_crew_tree)
                    self.console.print()

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_call_failed(source, event: LLMCallFailedEvent):
            if self.verbose:
                error_content = Text()
                error_content.append("❌ LLM Call Failed\n", style="red bold")
                error_content.append("Error: ", style="white")
                error_content.append(str(event.error), style="red")

                # Update under the agent branch if it exists
                if self.current_tool_branch:
                    self.current_tool_branch.label = Text(
                        "❌ LLM Failed", style="red bold"
                    )
                    self.console.print(self.current_crew_tree)
                    self.console.print()

                # Show error panel
                panel = self._create_panel(error_content, "LLM Error", "red")
                self.console.print(panel)
                self.console.print()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event: LLMStreamChunkEvent):
            self.text_stream.write(event.chunk)

            self.text_stream.seek(self.next_chunk)

            # Read from the in-memory stream
            content = self.text_stream.read()
            print(content, end="", flush=True)
            self.next_chunk = self.text_stream.tell()

        @crewai_event_bus.on(CrewTestStartedEvent)
        def on_crew_test_started(source, event: CrewTestStartedEvent):
            cloned_crew = source.copy()
            self._telemetry.test_execution_span(
                cloned_crew,
                event.n_iterations,
                event.inputs,
                event.eval_llm or "",
            )

            if self.verbose:
                content = Text()
                content.append("🧪 Starting Crew Test\n\n", style="blue bold")
                content.append("Crew: ", style="white")
                content.append(f"{event.crew_name}\n", style="blue")
                content.append("ID: ", style="white")
                content.append(str(source.id), style="blue")
                content.append("\nIterations: ", style="white")
                content.append(str(event.n_iterations), style="yellow")

                panel = self._create_panel(content, "Test Execution", "blue")

                self.console.print()
                self.console.print(panel)
                self.console.print()

                # Create and display the test tree
                test_label = Text()
                test_label.append("🧪 Test: ", style="blue bold")
                test_label.append(event.crew_name or "Crew", style="blue")
                test_label.append("\n    Status: ", style="white")
                test_label.append("In Progress", style="yellow")

                self.current_flow_tree = Tree(test_label)
                self.current_flow_tree.add(Text("🔄 Running tests...", style="yellow"))

                self.console.print(self.current_flow_tree)
                self.console.print()

        @crewai_event_bus.on(CrewTestCompletedEvent)
        def on_crew_test_completed(source, event: CrewTestCompletedEvent):
            if self.verbose:
                if self.current_flow_tree:
                    # Update test tree label to show completion
                    test_label = Text()
                    test_label.append("✅ Test: ", style="green bold")
                    test_label.append(event.crew_name or "Crew", style="green")
                    test_label.append("\n    Status: ", style="white")
                    test_label.append("Completed", style="green bold")
                    self.current_flow_tree.label = test_label

                    # Update the running tests node
                    for child in self.current_flow_tree.children:
                        if "Running tests" in str(child.label):
                            child.label = Text(
                                "✅ Tests completed successfully", style="green"
                            )

                    self.console.print(self.current_flow_tree)
                    self.console.print()

                # Create completion panel
                completion_content = Text()
                completion_content.append(
                    "Test Execution Completed\n", style="green bold"
                )
                completion_content.append("Crew: ", style="white")
                completion_content.append(f"{event.crew_name}\n", style="green")
                completion_content.append("Status: ", style="white")
                completion_content.append("All tests passed", style="green")

                panel = self._create_panel(
                    completion_content, "Test Completion", "green"
                )
                self.console.print(panel)
                self.console.print()


event_listener = EventListener()

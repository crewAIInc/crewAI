import threading
from typing import Any, ClassVar

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class ConsoleFormatter:
    tool_usage_counts: ClassVar[dict[str, int]] = {}

    current_a2a_turn_count: int = 0
    _pending_a2a_message: str | None = None
    _pending_a2a_agent_role: str | None = None
    _pending_a2a_turn_number: int | None = None
    _current_a2a_agent_name: str | None = None

    crew_completion_printed: ClassVar[threading.Event] = threading.Event()

    def __init__(self, verbose: bool = False):
        self.console = Console(width=None)
        self.verbose = verbose
        self._streaming_live: Live | None = None
        self._is_streaming: bool = False
        self._just_streamed_final_answer: bool = False
        self._last_stream_call_type: Any = None

    def create_panel(self, content: Text, title: str, style: str = "blue") -> Panel:
        """Create a standardized panel with consistent styling."""
        return Panel(
            content,
            title=title,
            border_style=style,
            padding=(1, 2),
        )

    def _show_tracing_disabled_message_if_needed(self) -> None:
        """Show tracing disabled message if tracing is not enabled."""
        from crewai.events.listeners.tracing.utils import (
            has_user_declined_tracing,
            is_tracing_enabled_in_context,
        )

        if not is_tracing_enabled_in_context():
            if has_user_declined_tracing():
                message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Crew/Flow code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""
            else:
                message = """Info: Tracing is disabled.

To enable tracing, do any one of these:
• Set tracing=True in your Crew/Flow code
• Set CREWAI_TRACING_ENABLED=true in your project's .env file
• Run: crewai traces enable"""

            panel = Panel(
                message,
                title="Tracing Status",
                border_style="blue",
                padding=(1, 2),
            )
            self.console.print(panel)

    def create_status_content(
        self,
        title: str,
        name: str,
        status_style: str = "blue",
        tool_args: dict[str, Any] | str = "",
        **fields: Any,
    ) -> Text:
        """Create standardized status content with consistent formatting."""
        content = Text()
        content.append(f"{title}\n", style=f"{status_style} bold")
        content.append("Name: \n", style="white")
        content.append(f"{name}\n", style=status_style)

        for label, value in fields.items():
            content.append(f"{label}: \n", style="white")
            content.append(
                f"{value}\n", style=fields.get(f"{label}_style", status_style)
            )
        if tool_args:
            content.append("Tool Args: \n", style="white")
            content.append(f"{tool_args}\n", style=status_style)

        return content

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to console. Simplified to only handle panel-based output."""
        # Skip blank lines during streaming
        if len(args) == 0 and self._is_streaming:
            return
        self.console.print(*args, **kwargs)

    def pause_live_updates(self) -> None:
        """Pause Live session updates to allow for human input without interference.

        This stops any active streaming Live session to prevent console refresh
        interference during HITL (Human-in-the-Loop) user input.
        """
        if self._streaming_live:
            self._streaming_live.stop()
            self._streaming_live = None

    def resume_live_updates(self) -> None:
        """Resume Live session updates after human input is complete.

        New streaming sessions will be created on-demand when needed.
        This method exists for API compatibility with HITL callers.
        """

    def print_panel(
        self, content: Text, title: str, style: str = "blue", is_flow: bool = False
    ) -> None:
        """Print a panel with consistent formatting if verbose is enabled."""
        panel = self.create_panel(content, title, style)
        if is_flow:
            self.print(panel)
            self.print()
        else:
            if self.verbose:
                self.print(panel)
                self.print()

    def handle_crew_status(
        self,
        crew_name: str,
        source_id: str,
        status: str = "completed",
        final_string_output: str = "",
    ) -> None:
        """Handle crew completion/failure with panel display."""
        if not self.verbose:
            return

        if status == "completed":
            style = "green"
            title = "Crew Completion"
            content_title = "Crew Execution Completed"
        elif status == "failed":
            style = "red"
            title = "Crew Failure"
            content_title = "Crew Execution Failed"
        else:
            style = "cyan"
            title = "Crew Execution"
            content_title = "Crew Execution Started"

        content = self.create_status_content(
            content_title,
            crew_name or "Crew",
            style,
            ID=source_id,
        )

        if status == "failed" and final_string_output:
            content.append("Error:\n", style="white bold")
            content.append(f"{final_string_output}\n", style="red")
        elif final_string_output:
            content.append(f"Final Output: {final_string_output}\n", style="white")

        self.print_panel(content, title, style)

        if status in ["completed", "failed"]:
            self.crew_completion_printed.set()
            self._show_tracing_disabled_message_if_needed()

    def handle_crew_started(self, crew_name: str, source_id: str) -> None:
        """Show crew started panel."""
        if not self.verbose:
            return

        # Reset the crew completion event for this new crew execution
        ConsoleFormatter.crew_completion_printed.clear()

        content = self.create_status_content(
            "Crew Execution Started",
            crew_name,
            "cyan",
            ID=source_id,
        )

        self.print_panel(content, "🚀 Crew Execution Started", "cyan")

    def handle_task_started(self, task_id: str, task_name: str | None = None) -> None:
        """Show task started panel."""
        if not self.verbose:
            return

        content = Text()
        display_name = task_name if task_name else task_id

        content.append("Task Started\n", style="yellow bold")
        content.append("Name: ", style="white")
        content.append(f"{display_name}\n", style="yellow")
        content.append("ID: ", style="white")
        content.append(f"{task_id}\n", style="yellow ")

        self.print_panel(content, "📋 Task Started", "yellow")

    def handle_task_status(
        self,
        task_id: str,
        agent_role: str,
        status: str = "completed",
        task_name: str | None = None,
    ) -> None:
        """Show task completion/failure panel."""
        if not self.verbose:
            return

        if status == "completed":
            style = "green"
            panel_title = "📋 Task Completion"
        else:
            style = "red"
            panel_title = "📋 Task Failure"

        display_name = task_name if task_name else str(task_id)
        content = self.create_status_content(
            f"Task {status.title()}", display_name, style, Agent=agent_role
        )
        self.print_panel(content, panel_title, style)

    def handle_flow_created(self, flow_name: str, flow_id: str) -> None:
        """Show flow started panel."""
        content = self.create_status_content(
            "Starting Flow Execution", flow_name, "blue", ID=flow_id
        )
        self.print_panel(content, "🌊 Flow Execution", "blue", is_flow=True)

    def handle_flow_started(self, flow_name: str, flow_id: str) -> None:
        """Show flow started panel."""
        content = Text()
        content.append("Flow Started\n", style="blue bold")
        content.append("Name: ", style="white")
        content.append(f"{flow_name}\n", style="blue")
        content.append("ID: ", style="white")
        content.append(f"{flow_id}\n", style="blue")

        self.print_panel(content, "🌊 Flow Started", "blue", is_flow=True)

    def handle_flow_status(
        self,
        flow_name: str,
        flow_id: str,
        status: str = "completed",
    ) -> None:
        """Show flow status panel."""
        if status == "completed":
            style = "green"
            content_text = "Flow Execution Completed"
            panel_title = "✅ Flow Completion"
        elif status == "paused":
            style = "cyan"
            content_text = "Flow Paused - Waiting for Feedback"
            panel_title = "⏳ Flow Paused"
        else:
            style = "red"
            content_text = "Flow Execution Failed"
            panel_title = "❌ Flow Failure"

        content = self.create_status_content(
            content_text,
            flow_name,
            style,
            ID=flow_id,
        )
        self.print_panel(content, panel_title, style, is_flow=True)

    def handle_method_status(
        self,
        method_name: str,
        status: str = "running",
    ) -> None:
        """Show method status panel."""
        if not self.verbose:
            return

        if status == "running":
            style = "yellow"
            panel_title = "🔄 Flow Method Running"
        elif status == "completed":
            style = "green"
            panel_title = "✅ Flow Method Completed"
        elif status == "paused":
            style = "cyan"
            panel_title = "⏳ Flow Method Paused"
        else:
            style = "red"
            panel_title = "❌ Flow Method Failed"

        content = Text()
        content.append(f"Method: {method_name}\n", style=f"{style} bold")
        content.append("Status: ", style="white")
        content.append(f"{status.title()}\n", style=style)

        self.print_panel(content, panel_title, style, is_flow=True)

    def handle_llm_tool_usage_started(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | str,
    ) -> None:
        """Handle LLM tool usage started with panel display."""
        content = self.create_status_content(
            "Tool Usage Started", tool_name, Status="In Progress", tool_args=tool_args
        )
        self.print_panel(content, "🔧 LLM Tool Usage", "yellow")

    def handle_llm_tool_usage_finished(
        self,
        tool_name: str,
    ) -> None:
        """Handle LLM tool usage finished with panel display."""
        content = Text()
        content.append("Tool Usage Completed\n", style="green bold")
        content.append("Tool: ", style="white")
        content.append(f"{tool_name}\n", style="green")

        self.print_panel(content, "✅ LLM Tool Completed", "green")

    def handle_llm_tool_usage_error(
        self,
        tool_name: str,
        error: str,
    ) -> None:
        """Handle LLM tool usage error with panel display."""
        error_content = self.create_status_content(
            "Tool Usage Failed", tool_name, "red", Error=error
        )
        self.print_panel(error_content, "❌ LLM Tool Error", "red")

    def handle_tool_usage_started(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | str = "",
        run_attempts: int | None = None,
    ) -> None:
        """Handle tool usage started event with panel display."""
        if not self.verbose:
            return

        # Update tool usage count
        self.tool_usage_counts[tool_name] = self.tool_usage_counts.get(tool_name, 0) + 1
        iteration = self.tool_usage_counts[tool_name]

        content = Text()
        content.append("Tool: ", style="white")
        content.append(f"{tool_name}\n", style="yellow bold")

        if tool_args:
            content.append("Args: ", style="white")
            args_str = (
                str(tool_args)[:200] + "..."
                if len(str(tool_args)) > 200
                else str(tool_args)
            )
            content.append(f"{args_str}\n", style="yellow ")

        self.print_panel(content, f"🔧 Tool Execution Started (#{iteration})", "yellow")

    def handle_tool_usage_finished(
        self,
        tool_name: str,
        output: str,
        run_attempts: int | None = None,
    ) -> None:
        """Handle tool usage finished event with panel display."""
        if not self.verbose:
            return

        iteration = self.tool_usage_counts.get(tool_name, 1)

        content = Text()
        content.append("Tool Completed\n", style="green bold")
        content.append("Tool: ", style="white")
        content.append(f"{tool_name}\n", style="green bold")

        if output:
            content.append("Output: ", style="white")

            content.append(f"{output}\n", style="green")

        self.print_panel(
            content, f"✅ Tool Execution Completed (#{iteration})", "green"
        )

    def handle_tool_usage_error(
        self,
        tool_name: str,
        error: str,
        run_attempts: int | None = None,
    ) -> None:
        """Handle tool usage error event with panel display."""
        if not self.verbose:
            return

        iteration = self.tool_usage_counts.get(tool_name, 1)

        content = Text()
        content.append("Tool Failed\n", style="red bold")
        content.append("Tool: ", style="white")
        content.append(f"{tool_name}\n", style="red bold")
        content.append("Iteration: ", style="white")
        content.append(f"{iteration}\n", style="red")
        if run_attempts is not None:
            content.append("Attempt: ", style="white")
            content.append(f"{run_attempts}\n", style="red")
        content.append("Error: ", style="white")
        content.append(f"{error}\n", style="red")

        self.print_panel(content, f"🔧 Tool Error (#{iteration})", "red")

    def handle_llm_call_failed(self, error: str) -> None:
        """Handle LLM call failed event with panel display."""
        if not self.verbose:
            return

        error_content = Text()
        error_content.append("LLM Call Failed\n", style="red bold")
        error_content.append("Error: ", style="white")
        error_content.append(str(error), style="red")

        self.print_panel(error_content, "❌ LLM Error", "red")

    def handle_llm_stream_chunk(
        self,
        accumulated_text: str,
        call_type: Any = None,
    ) -> None:
        """Handle LLM stream chunk event - display streaming text in a panel.

        Args:
            chunk: The new chunk of text received.
            accumulated_text: All text accumulated so far.
            crew_tree: Unused (kept for API compatibility).
            call_type: The type of LLM call (LLM_CALL or TOOL_CALL).
        """
        if not self.verbose:
            return

        self._is_streaming = True
        self._last_stream_call_type = call_type

        display_text = accumulated_text
        max_lines = 20
        lines = display_text.split("\n")
        if len(lines) > max_lines:
            display_text = "\n".join(lines[-max_lines:])
            display_text = "...\n" + display_text

        content = Text()

        from crewai.events.types.llm_events import LLMCallType

        if call_type == LLMCallType.TOOL_CALL:
            content.append(display_text, style="yellow")
            title = "🔧 Tool Arguments"
            border_style = "yellow"
        else:
            content.append(display_text, style="bright_green")
            title = "✅ Agent Final Answer"
            border_style = "green"

        streaming_panel = Panel(
            content,
            title=title,
            border_style=border_style,
            padding=(1, 2),
        )

        if not self._streaming_live:
            self._streaming_live = Live(
                streaming_panel, console=self.console, refresh_per_second=10
            )
            self._streaming_live.start()
        else:
            self._streaming_live.update(streaming_panel, refresh=True)

    def handle_llm_stream_completed(self) -> None:
        """Handle completion of LLM streaming - stop the streaming live display."""
        self._is_streaming = False

        from crewai.events.types.llm_events import LLMCallType

        if self._last_stream_call_type == LLMCallType.LLM_CALL:
            self._just_streamed_final_answer = True
        else:
            self._just_streamed_final_answer = False

        self._last_stream_call_type = None

        if self._streaming_live:
            self._streaming_live.stop()
            self._streaming_live = None

    def handle_crew_test_started(
        self, crew_name: str, source_id: str, n_iterations: int
    ) -> None:
        """Handle crew test started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Starting Crew Test\n", style="blue bold")
        content.append("Crew: ", style="white")
        content.append(f"{crew_name}\n", style="blue")
        content.append("ID: ", style="white")
        content.append(f"{source_id}\n", style="blue")
        content.append("Iterations: ", style="white")
        content.append(f"{n_iterations}\n", style="yellow")
        content.append("Status: ", style="white")
        content.append("Running...", style="yellow")

        self.print_panel(content, "🧪 Test Execution Started", "blue")

    def handle_crew_test_completed(self, crew_name: str) -> None:
        """Handle crew test completed event with panel display."""
        if not self.verbose:
            return

        completion_content = Text()
        completion_content.append("Test Execution Completed\n", style="green bold")
        completion_content.append("Crew: ", style="white")
        completion_content.append(f"{crew_name}\n", style="green")
        completion_content.append("\nStatus: ", style="white")
        completion_content.append("Completed", style="green")

        self.print_panel(completion_content, "Test Completion", "green")

    def handle_crew_train_started(self, crew_name: str, timestamp: str) -> None:
        """Handle crew train started event."""
        if not self.verbose:
            return

        content = Text()
        content.append("📋 Crew Training Started\n", style="blue bold")
        content.append("Crew: ", style="white")
        content.append(f"{crew_name}\n", style="blue")
        content.append("Time: ", style="white")
        content.append(timestamp, style="blue")

        self.print_panel(content, "Training Started", "blue")
        self.print()

    def handle_crew_train_completed(self, crew_name: str, timestamp: str) -> None:
        """Handle crew train completed event."""
        if not self.verbose:
            return

        content = Text()
        content.append("✅ Crew Training Completed\n", style="green bold")
        content.append("Crew: ", style="white")
        content.append(f"{crew_name}\n", style="green")
        content.append("Time: ", style="white")
        content.append(timestamp, style="green")

        self.print_panel(content, "Training Completed", "green")
        self.print()

    def handle_crew_train_failed(self, crew_name: str) -> None:
        """Handle crew train failed event."""
        if not self.verbose:
            return

        failure_content = Text()
        failure_content.append("❌ Crew Training Failed\n", style="red bold")
        failure_content.append("Crew: ", style="white")
        failure_content.append(crew_name or "Crew", style="red")

        self.print_panel(failure_content, "Training Failure", "red")
        self.print()

    def handle_crew_test_failed(self, crew_name: str) -> None:
        """Handle crew test failed event."""
        if not self.verbose:
            return

        failure_content = Text()
        failure_content.append("❌ Crew Test Failed\n", style="red bold")
        failure_content.append("Crew: ", style="white")
        failure_content.append(crew_name or "Crew", style="red")

        self.print_panel(failure_content, "Test Failure", "red")
        self.print()

    def create_lite_agent_branch(self, lite_agent_role: str) -> None:
        """Show lite agent started panel."""
        if not self.verbose:
            return

        content = Text()
        content.append("LiteAgent Started\n", style="cyan bold")
        content.append("Role: ", style="white")
        content.append(f"{lite_agent_role}\n", style="cyan")
        content.append("Status: ", style="white")
        content.append("In Progress\n", style="yellow")

        self.print_panel(content, "🤖 LiteAgent Started", "cyan")

    def update_lite_agent_status(
        self,
        lite_agent_role: str,
        status: str = "completed",
        **fields: dict[str, Any],
    ) -> None:
        """Show lite agent status panel."""
        if not self.verbose:
            return

        if status == "completed":
            style = "green"
            title = "✅ LiteAgent Completed"
        elif status == "failed":
            style = "red"
            title = "❌ LiteAgent Failed"
        else:
            style = "yellow"
            title = "🤖 LiteAgent Status"

        content = Text()
        content.append(f"LiteAgent {status.title()}\n", style=f"{style} bold")
        content.append("Role: ", style="white")
        content.append(f"{lite_agent_role}\n", style=style)

        for field_name, field_value in fields.items():
            content.append(f"{field_name}: ", style="white")
            content.append(f"{field_value}\n", style=style)

        self.print_panel(content, title, style)

    def handle_lite_agent_execution(
        self,
        lite_agent_role: str,
        status: str = "started",
        error: Any = None,
        **fields: dict[str, Any],
    ) -> None:
        """Handle lite agent execution events with panel display."""
        if not self.verbose:
            return

        if status == "started":
            self.create_lite_agent_branch(lite_agent_role)
            if fields:
                content = self.create_status_content(
                    "LiteAgent Session Started", lite_agent_role, "cyan", **fields
                )
                self.print_panel(content, "🤖 LiteAgent Started", "cyan")
        else:
            if error:
                fields["Error"] = error
            self.update_lite_agent_status(lite_agent_role, status, **fields)

    def handle_knowledge_retrieval_started(
        self,
    ) -> None:
        """Handle knowledge retrieval started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Knowledge Retrieval Started\n", style="blue bold")
        content.append("Status: ", style="white")
        content.append("Retrieving...\n", style="blue")

        self.print_panel(content, "🔍 Knowledge Retrieval", "blue")

    def handle_knowledge_retrieval_completed(
        self,
        retrieved_knowledge: Any,
        search_query: str,
    ) -> None:
        """Handle knowledge retrieval completed event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Search Query:\n", style="white")
        content.append(f"{search_query}\n", style="green")
        content.append("Knowledge Retrieved: \n", style="white")
        if retrieved_knowledge:
            knowledge_text = str(retrieved_knowledge)
            if len(knowledge_text) > 500:
                knowledge_text = knowledge_text[:497] + "..."
            content.append(f"{knowledge_text}\n", style="green ")
        else:
            content.append("No knowledge retrieved\n", style="yellow")

        self.print_panel(content, "📚 Knowledge Retrieved", "green")

    def handle_knowledge_query_started(
        self,
        task_prompt: str,
    ) -> None:
        """Handle knowledge query started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Knowledge Query Started\n", style="yellow bold")
        content.append("Query: ", style="white")
        query_preview = (
            task_prompt[:100] + "..." if len(task_prompt) > 100 else task_prompt
        )
        content.append(f"{query_preview}\n", style="yellow")

        self.print_panel(content, "🔎 Knowledge Query", "yellow")

    def handle_knowledge_query_failed(
        self,
        error: str,
    ) -> None:
        """Handle knowledge query failed event with panel display."""
        if not self.verbose:
            return

        error_content = self.create_status_content(
            "Knowledge Query Failed", "Query Error", "red", Error=error
        )
        self.print_panel(error_content, "❌ Knowledge Error", "red")

    def handle_knowledge_query_completed(self) -> None:
        """Handle knowledge query completed event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Knowledge Query Completed\n", style="green bold")

        self.print_panel(content, "✅ Knowledge Query Complete", "green")

    def handle_knowledge_search_query_failed(
        self,
        error: str,
    ) -> None:
        """Handle knowledge search query failed event with panel display."""
        if not self.verbose:
            return

        error_content = self.create_status_content(
            "Knowledge Search Failed", "Search Error", "red", Error=error
        )
        self.print_panel(error_content, "❌ Search Error", "red")

    # ----------- AGENT REASONING EVENTS -----------

    def handle_reasoning_started(
        self,
        attempt: int,
    ) -> None:
        """Handle agent reasoning started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Reasoning Started\n", style="blue bold")
        content.append("Attempt: ", style="white")
        content.append(f"{attempt}\n", style="blue")
        content.append("Status: ", style="white")
        content.append("Thinking...\n", style="blue")

        panel_title = (
            f"🧠 Reasoning (Attempt #{attempt})" if attempt > 1 else "🧠 Reasoning"
        )
        self.print_panel(content, panel_title, "blue")

    def handle_reasoning_completed(
        self,
        plan: str,
        ready: bool,
    ) -> None:
        """Handle agent reasoning completed event with panel display."""
        if not self.verbose:
            return

        style = "green" if ready else "yellow"
        status_text = "Ready" if ready else "Not Ready"

        content = Text()
        content.append("Reasoning Completed\n", style=f"{style} bold")
        content.append("Status: ", style="white")
        content.append(f"{status_text}\n", style=style)

        if plan:
            plan_preview = plan[:500] + "..." if len(plan) > 500 else plan
            content.append("Plan: ", style="white")
            content.append(f"{plan_preview}\n", style=style)

        self.print_panel(content, "✅ Reasoning Complete", style)

    def handle_reasoning_failed(
        self,
        error: str,
    ) -> None:
        """Handle agent reasoning failure event with panel display."""
        if not self.verbose:
            return

        error_content = self.create_status_content(
            "Reasoning Failed",
            "Error",
            "red",
            Error=error,
        )
        self.print_panel(error_content, "❌ Reasoning Error", "red")

    # ----------- AGENT LOGGING EVENTS -----------

    def handle_agent_logs_started(
        self,
        agent_role: str,
        task_description: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Handle agent logs started event."""
        if not verbose:
            return

        agent_role = agent_role.partition("\n")[0]

        # Create panel content
        content = Text()
        content.append("Agent: ", style="white")
        content.append(f"{agent_role}", style="bright_green bold")

        if task_description:
            content.append("\n\nTask: ", style="white")
            content.append(f"{task_description}", style="bright_green")

        # Create and display the panel
        agent_panel = Panel(
            content,
            title="🤖 Agent Started",
            border_style="magenta",
            padding=(1, 2),
        )
        self.print(agent_panel)
        self.print()

    def handle_agent_logs_execution(
        self,
        agent_role: str,
        formatted_answer: Any,
        verbose: bool = False,
    ) -> None:
        """Handle agent logs execution event."""
        if not verbose:
            return

        import json

        from crewai.agents.parser import AgentAction, AgentFinish

        agent_role = agent_role.partition("\n")[0]

        if isinstance(formatted_answer, AgentAction):
            # Create tool output content with better formatting
            output_text = str(formatted_answer.result)
            if len(output_text) > 2000:
                output_text = output_text[:1997] + "..."

            output_panel = Panel(
                Text(output_text, style="bright_green"),
                title="Tool Output",
                border_style="green",
                padding=(1, 2),
            )

            # Print all panels
            self.print(output_panel)
            self.print()

        elif isinstance(formatted_answer, AgentFinish):
            if self._just_streamed_final_answer:
                self._just_streamed_final_answer = False
                return

            is_a2a_delegation = False
            try:
                output_data = json.loads(formatted_answer.output)
                if isinstance(output_data, dict):
                    if output_data.get("is_a2a") is True:
                        is_a2a_delegation = True
                    elif "output" in output_data:
                        nested_output = output_data["output"]
                        if (
                            isinstance(nested_output, dict)
                            and nested_output.get("is_a2a") is True
                        ):
                            is_a2a_delegation = True
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            if not is_a2a_delegation:
                content = Text()
                content.append("Agent: ", style="white")
                content.append(f"{agent_role}\n\n", style="bright_green bold")
                content.append("Final Answer:\n", style="white")
                content.append(f"{formatted_answer.output}", style="bright_green")

                finish_panel = Panel(
                    content,
                    title="✅ Agent Final Answer",
                    border_style="green",
                    padding=(1, 2),
                )
                self.print(finish_panel)
                self.print()

    def handle_memory_retrieval_started(self) -> None:
        """Handle memory retrieval started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Memory Retrieval Started\n", style="blue bold")
        content.append("Status: ", style="white")
        content.append("Retrieving...\n", style="blue")

        self.print_panel(content, "🧠 Memory Retrieval", "blue")

    def handle_memory_retrieval_completed(
        self,
        memory_content: str,
        retrieval_time_ms: float,
    ) -> None:
        """Handle memory retrieval completed event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Memory Retrieval Completed\n", style="green bold")
        content.append("Time: ", style="white")
        content.append(f"{retrieval_time_ms:.2f}ms\n", style="green")

        if memory_content:
            memory_text = str(memory_content)

            content.append("Content: \n", style="white")
            content.append(f"{memory_text}\n", style="green ")

        self.print_panel(content, "🧠 Memory Retrieved", "green")

    def handle_memory_query_failed(
        self,
        error: str,
        source_type: str,
    ) -> None:
        """Handle memory query failed event with panel display."""
        if not self.verbose:
            return

        memory_type = source_type.replace("_", " ").title()

        content = Text()
        content.append("Memory Query Failed\n", style="red bold")
        content.append("Source: ", style="white")
        content.append(f"{memory_type}\n", style="red")
        content.append("Error: ", style="white")
        content.append(f"{error}\n", style="red")

        self.print_panel(content, "❌ Memory Query Error", "red")

    def handle_memory_save_started(self) -> None:
        """Handle memory save started event with panel display."""
        if not self.verbose:
            return

        content = Text()
        content.append("Memory Save Started\n", style="blue bold")
        content.append("Status: ", style="white")
        content.append("Saving...\n", style="blue")

        self.print_panel(content, "🧠 Memory Save", "blue")

    def handle_memory_save_completed(
        self,
        save_time_ms: float,
        source_type: str,
    ) -> None:
        """Handle memory save completed event with panel display."""
        if not self.verbose:
            return

        memory_type = source_type.replace("_", " ").title()

        content = Text()
        content.append("Memory Save Completed\n", style="green bold")
        content.append("Source: ", style="white")
        content.append(f"{memory_type}\n", style="green")
        content.append("Time: ", style="white")
        content.append(f"{save_time_ms:.2f}ms\n", style="green")

        self.print_panel(content, "✅ Memory Saved", "green")

    def handle_memory_save_failed(
        self,
        error: str,
        source_type: str,
    ) -> None:
        """Handle memory save failed event with panel display."""
        if not self.verbose:
            return

        memory_type = source_type.replace("_", " ").title()

        content = Text()
        content.append("Memory Save Failed\n", style="red bold")
        content.append("Source: ", style="white")
        content.append(f"{memory_type}\n", style="red")
        content.append("Error: ", style="white")
        content.append(f"{error}\n", style="red")

        self.print_panel(content, "❌ Memory Save Error", "red")

    def handle_guardrail_started(
        self,
        guardrail_name: str,
        retry_count: int,
    ) -> None:
        """Display guardrail evaluation started status.

        Args:
            guardrail_name: Name/description of the guardrail being evaluated.
            retry_count: Zero-based retry count (0 = first attempt).
        """
        if not self.verbose:
            return

        content = self.create_status_content(
            "Guardrail Evaluation Started",
            guardrail_name,
            "yellow",
            Status="🔄 Evaluating",
            Attempt=f"{retry_count + 1}",
        )
        self.print_panel(content, "🛡️ Guardrail Check", "yellow")

    def handle_guardrail_completed(
        self,
        success: bool,
        error: str | None,
        retry_count: int,
    ) -> None:
        """Display guardrail evaluation result.

        Args:
            success: Whether validation passed.
            error: Error message if validation failed.
            retry_count: Zero-based retry count.
        """
        if not self.verbose:
            return

        if success:
            content = self.create_status_content(
                "Guardrail Passed",
                "Validation Successful",
                "green",
                Status="✅ Validated",
                Attempts=f"{retry_count + 1}",
            )
            self.print_panel(content, "🛡️ Guardrail Success", "green")
        else:
            content = self.create_status_content(
                "Guardrail Failed",
                "Validation Error",
                "red",
                Error=str(error) if error else "Unknown error",
                Attempts=f"{retry_count + 1}",
            )
            self.print_panel(content, "🛡️ Guardrail Failed", "red")

    def handle_a2a_delegation_started(
        self,
        endpoint: str,
        task_description: str,
        agent_id: str,
        is_multiturn: bool = False,
        turn_number: int = 1,
    ) -> None:
        """Handle A2A delegation started event with panel display."""
        if is_multiturn:
            self.current_a2a_turn_count = turn_number

        content = Text()
        content.append("A2A Delegation Started\n", style="cyan bold")
        content.append("Agent ID: ", style="white")
        content.append(f"{agent_id}\n", style="cyan")
        content.append("Endpoint: ", style="white")
        content.append(f"{endpoint}\n", style="cyan")
        if is_multiturn:
            content.append("Turn: ", style="white")
            content.append(f"{turn_number}\n", style="cyan")
        content.append("Task: ", style="white")
        task_preview = (
            task_description[:200] + "..."
            if len(task_description) > 200
            else task_description
        )
        content.append(f"{task_preview}\n", style="cyan")

        self.print_panel(content, "🔗 A2A Delegation", "cyan")
        return

    def handle_a2a_delegation_completed(
        self,
        status: str,
        result: str | None = None,
        error: str | None = None,
        is_multiturn: bool = False,
    ) -> None:
        """Handle A2A delegation completed event with panel display."""
        if is_multiturn:
            if status in ["completed", "failed"]:
                self.current_a2a_turn_count = 0

        if status == "completed" and result:
            content = Text()
            content.append("A2A Delegation Completed\n", style="green bold")
            content.append("Result: ", style="white")
            result_preview = result[:500] + "..." if len(result) > 500 else result
            content.append(f"{result_preview}\n", style="green")

            self.print_panel(content, "✅ A2A Success", "green")
        elif status == "input_required" and error:
            content = Text()
            content.append("A2A Response Received\n", style="cyan bold")
            content.append("Message: ", style="white")
            response_preview = error[:500] + "..." if len(error) > 500 else error
            content.append(f"{response_preview}\n", style="cyan")

            self.print_panel(content, "💬 A2A Response", "cyan")
        elif status == "failed":
            content = Text()
            content.append("A2A Delegation Failed\n", style="red bold")
            if error:
                content.append("Error: ", style="white")
                content.append(f"{error}\n", style="red")

            self.print_panel(content, "❌ A2A Failed", "red")
        else:
            content = Text()
            content.append(f"A2A Delegation {status.title()}\n", style="yellow bold")
            if error:
                content.append("Message: ", style="white")
                content.append(f"{error}\n", style="yellow")

            self.print_panel(content, "⚠️ A2A Status", "yellow")

    def handle_a2a_conversation_started(
        self,
        agent_id: str,
        endpoint: str,
    ) -> None:
        """Handle A2A conversation started event with panel display."""
        content = Text()
        content.append("A2A Conversation Started\n", style="cyan bold")
        content.append("Agent ID: ", style="white")
        content.append(f"{agent_id}\n", style="cyan")
        content.append("Endpoint: ", style="white")
        content.append(f"{endpoint}\n", style="cyan ")

        self.print_panel(content, "💬 A2A Conversation", "cyan")

    def handle_a2a_message_sent(
        self,
        message: str,
        turn_number: int,
        agent_role: str | None = None,
    ) -> None:
        """Handle A2A message sent event - store for display with response."""
        self._pending_a2a_message = message
        self._pending_a2a_agent_role = agent_role
        self._pending_a2a_turn_number = turn_number

    def handle_a2a_response_received(
        self,
        response: str,
        turn_number: int,
        status: str,
        agent_role: str | None = None,
    ) -> None:
        """Handle A2A response received event with panel display."""
        crewai_agent_role = self._pending_a2a_agent_role or agent_role or "User"
        message_content = self._pending_a2a_message or ""

        # Determine status styling
        if status == "completed":
            style = "green"
            status_indicator = "✓"
        elif status == "input_required":
            style = "yellow"
            status_indicator = "❓"
        elif status == "failed":
            style = "red"
            status_indicator = "✗"
        elif status == "auth_required":
            style = "magenta"
            status_indicator = "🔒"
        elif status == "canceled":
            style = ""
            status_indicator = "⊘"
        else:
            style = "cyan"
            status_indicator = ""

        content = Text()
        content.append(f"A2A Turn {turn_number}\n", style="cyan bold")
        content.append("Status: ", style="white")
        content.append(f"{status_indicator} {status}\n", style=style)

        if message_content:
            content.append(f"{crewai_agent_role}: ", style="blue bold")
            msg_preview = (
                message_content[:200] + "..."
                if len(message_content) > 200
                else message_content
            )
            content.append(f"{msg_preview}\n", style="blue")

        content.append(
            f"{self._current_a2a_agent_name or 'A2A Agent'}: ", style=f"{style} bold"
        )
        response_preview = response[:200] + "..." if len(response) > 200 else response
        content.append(f"{response_preview}\n", style=style)

        self.print_panel(content, f"💬 A2A Turn #{turn_number}", style)

        # Clear pending state
        self._pending_a2a_message = None
        self._pending_a2a_agent_role = None
        self._pending_a2a_turn_number = None

    def handle_a2a_conversation_completed(
        self,
        status: str,
        final_result: str | None,
        error: str | None,
        total_turns: int,
    ) -> None:
        """Handle A2A conversation completed event with panel display."""
        if status == "completed":
            content = Text()
            content.append("A2A Conversation Completed\n", style="green bold")
            content.append("Total Turns: ", style="white")
            content.append(f"{total_turns}\n", style="green")
            if final_result:
                content.append("Result: ", style="white")
                result_preview = (
                    final_result[:500] + "..."
                    if len(final_result) > 500
                    else final_result
                )
                content.append(f"{result_preview}\n", style="green")

            self.print_panel(content, "✅ A2A Complete", "green")
        elif status == "failed":
            content = Text()
            content.append("A2A Conversation Failed\n", style="red bold")
            content.append("Total Turns: ", style="white")
            content.append(f"{total_turns}\n", style="red")
            if error:
                content.append("Error: ", style="white")
                content.append(f"{error}\n", style="red")

            self.print_panel(content, "❌ A2A Failed", "red")

        # Reset state
        self.current_a2a_turn_count = 0
        self._pending_a2a_message = None
        self._pending_a2a_agent_role = None
        self._pending_a2a_turn_number = None

    # ----------- MCP EVENTS -----------

    def handle_mcp_connection_started(
        self,
        server_name: str,
        server_url: str | None = None,
        transport_type: str | None = None,
        is_reconnect: bool = False,
        connect_timeout: int | None = None,
    ) -> None:
        """Handle MCP connection started event."""
        if not self.verbose:
            return

        content = Text()
        reconnect_text = " (Reconnecting)" if is_reconnect else ""
        content.append(f"MCP Connection Started{reconnect_text}\n\n", style="cyan bold")

        if server_url:
            content.append("URL: ", style="white")
            content.append(f"{server_url}\n", style="cyan ")

        if transport_type:
            content.append("Transport: ", style="white")
            content.append(f"{transport_type}\n", style="cyan")

        if connect_timeout:
            content.append("Timeout: ", style="white")
            content.append(f"{connect_timeout}s\n", style="cyan")

        panel = self.create_panel(content, "🔌 MCP Connection", "cyan")
        self.print(panel)
        self.print()

    def handle_mcp_connection_completed(
        self,
        server_name: str,
        server_url: str | None = None,
        transport_type: str | None = None,
        connection_duration_ms: float | None = None,
        is_reconnect: bool = False,
    ) -> None:
        """Handle MCP connection completed event."""
        if not self.verbose:
            return

        content = Text()
        reconnect_text = " (Reconnected)" if is_reconnect else ""
        content.append(
            f"MCP Connection Completed{reconnect_text}\n\n", style="green bold"
        )
        content.append("Server: ", style="white")
        content.append(f"{server_name}\n", style="green")

        if server_url:
            content.append("URL: ", style="white")
            content.append(f"{server_url}\n", style="green ")

        if transport_type:
            content.append("Transport: ", style="white")
            content.append(f"{transport_type}\n", style="green")

        if connection_duration_ms is not None:
            content.append("Duration: ", style="white")
            content.append(f"{connection_duration_ms:.2f}ms\n", style="green")

        panel = self.create_panel(content, "✅ MCP Connected", "green")
        self.print(panel)
        self.print()

    def handle_mcp_connection_failed(
        self,
        server_name: str,
        server_url: str | None = None,
        transport_type: str | None = None,
        error: str = "",
        error_type: str | None = None,
    ) -> None:
        """Handle MCP connection failed event."""
        if not self.verbose:
            return

        content = Text()
        content.append("MCP Connection Failed\n\n", style="red bold")
        content.append("Server: ", style="white")
        content.append(f"{server_name}\n", style="red")

        if server_url:
            content.append("URL: ", style="white")
            content.append(f"{server_url}\n", style="red ")

        if transport_type:
            content.append("Transport: ", style="white")
            content.append(f"{transport_type}\n", style="red")

        if error_type:
            content.append("Error Type: ", style="white")
            content.append(f"{error_type}\n", style="red")

        if error:
            content.append("\nError: ", style="white bold")
            error_preview = error[:500] + "..." if len(error) > 500 else error
            content.append(f"{error_preview}\n", style="red")

        panel = self.create_panel(content, "❌ MCP Connection Failed", "red")
        self.print(panel)
        self.print()

    def handle_mcp_tool_execution_started(
        self,
        server_name: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        """Handle MCP tool execution started event."""
        if not self.verbose:
            return

        content = self.create_status_content(
            "MCP Tool Started",
            tool_name,
            "yellow",
            tool_args=tool_args or {},
            Server=server_name,
        )

        panel = self.create_panel(content, "🔧 MCP Tool Started", "yellow")
        self.print(panel)
        self.print()

    def handle_mcp_tool_execution_failed(
        self,
        server_name: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        error: str = "",
        error_type: str | None = None,
    ) -> None:
        """Handle MCP tool execution failed event."""
        if not self.verbose:
            return

        content = self.create_status_content(
            "MCP Tool Execution Failed",
            tool_name,
            "red",
            tool_args=tool_args or {},
            Server=server_name,
        )

        if error_type:
            content.append("Error Type: ", style="white")
            content.append(f"{error_type}\n", style="red")

        if error:
            content.append("\nError: ", style="white bold")
            error_preview = error[:500] + "..." if len(error) > 500 else error
            content.append(f"{error_preview}\n", style="red")

        panel = self.create_panel(content, "❌ MCP Tool Failed", "red")
        self.print(panel)
        self.print()

    def handle_a2a_polling_started(
        self,
        task_id: str,
        polling_interval: float,
        endpoint: str,
    ) -> None:
        """Handle A2A polling started event with panel display."""
        content = Text()
        content.append("A2A Polling Started\n", style="cyan bold")
        content.append("Task ID: ", style="white")
        content.append(f"{task_id[:8]}...\n", style="cyan")
        content.append("Interval: ", style="white")
        content.append(f"{polling_interval}s\n", style="cyan")

        self.print_panel(content, "⏳ A2A Polling", "cyan")

    def handle_a2a_polling_status(
        self,
        task_id: str,
        state: str,
        elapsed_seconds: float,
        poll_count: int,
    ) -> None:
        """Handle A2A polling status event with panel display."""
        if state == "completed":
            style = "green"
            status_indicator = "✓"
        elif state == "failed":
            style = "red"
            status_indicator = "✗"
        elif state == "working":
            style = "yellow"
            status_indicator = "⋯"
        else:
            style = "cyan"
            status_indicator = "•"

        content = Text()
        content.append(f"Poll #{poll_count}\n", style=f"{style} bold")
        content.append("Status: ", style="white")
        content.append(f"{status_indicator} {state}\n", style=style)
        content.append("Elapsed: ", style="white")
        content.append(f"{elapsed_seconds:.1f}s\n", style=style)

        self.print_panel(content, f"📊 A2A Poll #{poll_count}", style)

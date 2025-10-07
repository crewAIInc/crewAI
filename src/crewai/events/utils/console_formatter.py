from typing import Any, ClassVar

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree


class ConsoleFormatter:
    current_crew_tree: Tree | None = None
    current_task_branch: Tree | None = None
    current_agent_branch: Tree | None = None
    current_tool_branch: Tree | None = None
    current_flow_tree: Tree | None = None
    current_method_branch: Tree | None = None
    current_lite_agent_branch: Tree | None = None
    tool_usage_counts: ClassVar[dict[str, int]] = {}
    current_reasoning_branch: Tree | None = None  # Track reasoning status
    _live_paused: bool = False
    current_llm_tool_tree: Tree | None = None

    def __init__(self, verbose: bool = False):
        self.console = Console(width=None)
        self.verbose = verbose
        # Live instance to dynamically update a Tree renderable (e.g. the Crew tree)
        # When multiple Tree objects are printed sequentially we reuse this Live
        # instance so the previous render is replaced instead of writing a new one.
        # Once any non-Tree renderable is printed we stop the Live session so the
        # final Tree persists on the terminal.
        self._live: Live | None = None

    def create_panel(self, content: Text, title: str, style: str = "blue") -> Panel:
        """Create a standardized panel with consistent styling."""
        return Panel(
            content,
            title=title,
            border_style=style,
            padding=(1, 2),
        )

    def create_status_content(
        self,
        title: str,
        name: str,
        status_style: str = "blue",
        tool_args: dict[str, Any] | str = "",
        **fields,
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
        content.append("Tool Args: ", style="white")
        content.append(f"{tool_args}\n", style=status_style)

        return content

    def update_tree_label(
        self,
        tree: Tree,
        prefix: str,
        name: str,
        style: str = "blue",
        status: str | None = None,
    ) -> None:
        """Update tree label with consistent formatting."""
        label = Text()
        label.append(f"{prefix} ", style=f"{style} bold")
        label.append(name, style=style)
        if status:
            label.append("\nStatus: ", style="white")
            label.append(status, style=f"{style} bold")
        tree.label = label

    def add_tree_node(self, parent: Tree, text: str, style: str = "yellow") -> Tree:
        """Add a node to the tree with consistent styling."""
        return parent.add(Text(text, style=style))

    def print(self, *args, **kwargs) -> None:
        """Custom print that replaces consecutive Tree renders.

        * If the argument is a single ``Tree`` instance, we either start a
          ``Live`` session (first tree) or update the existing one (subsequent
          trees). This results in the tree being rendered in-place instead of
          being appended repeatedly to the log.

        * A blank call (no positional arguments) is ignored while a Live
          session is active so it does not prematurely terminate the tree
          rendering.

        * Any other renderable will terminate the Live session (if one is
          active) so the last tree stays on screen and the new content is
          printed normally.
        """

        # Case 1: updating / starting live Tree rendering
        if len(args) == 1 and isinstance(args[0], Tree):
            tree = args[0]

            if not self._live:
                # Start a new Live session for the first tree
                self._live = Live(tree, console=self.console, refresh_per_second=4)
                self._live.start()
            else:
                # Update existing Live session
                self._live.update(tree, refresh=True)
            return  # Nothing else to do

        # Case 2: blank line while a live session is running - ignore so we
        # don't break the in-place rendering behaviour
        if len(args) == 0 and self._live:
            return

        # Case 3: printing something other than a Tree → terminate live session
        if self._live:
            self._live.stop()
            self._live = None

        # Finally, pass through to the regular Console.print implementation
        self.console.print(*args, **kwargs)

    def pause_live_updates(self) -> None:
        """Pause Live session updates to allow for human input without interference."""
        if not self._live_paused:
            if self._live:
                self._live.stop()
                self._live = None
            self._live_paused = True

    def resume_live_updates(self) -> None:
        """Resume Live session updates after human input is complete."""
        if self._live_paused:
            self._live_paused = False

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

    def update_crew_tree(
        self,
        tree: Tree | None,
        crew_name: str,
        source_id: str,
        status: str = "completed",
        final_string_output: str = "",
    ) -> None:
        """Handle crew tree updates with consistent formatting."""
        if not self.verbose or tree is None:
            return

        if status == "completed":
            prefix, style = "✅ Crew:", "green"
            title = "Crew Completion"
            content_title = "Crew Execution Completed"
        elif status == "failed":
            prefix, style = "❌ Crew:", "red"
            title = "Crew Failure"
            content_title = "Crew Execution Failed"
        else:
            prefix, style = "🚀 Crew:", "cyan"
            title = "Crew Execution"
            content_title = "Crew Execution Started"

        self.update_tree_label(
            tree,
            prefix,
            crew_name or "Crew",
            style,
        )

        content = self.create_status_content(
            content_title,
            crew_name or "Crew",
            style,
            ID=source_id,
        )
        content.append(f"Final Output: {final_string_output}\n", style="white")

        self.print_panel(content, title, style)

    def create_crew_tree(self, crew_name: str, source_id: str) -> Tree | None:
        """Create and initialize a new crew tree with initial status."""
        if not self.verbose:
            return None

        tree = Tree(
            Text("🚀 Crew: ", style="cyan bold") + Text(crew_name, style="cyan")
        )

        content = self.create_status_content(
            "Crew Execution Started",
            crew_name,
            "cyan",
            ID=source_id,
        )

        self.print_panel(content, "Crew Execution Started", "cyan")

        # Set the current_crew_tree attribute directly
        self.current_crew_tree = tree

        return tree

    def create_task_branch(
        self, crew_tree: Tree | None, task_id: str, task_name: str | None = None
    ) -> Tree | None:
        """Create and initialize a task branch."""
        if not self.verbose:
            return None

        task_content = Text()

        # Display task name if available, otherwise just the ID
        if task_name:
            task_content.append("📋 Task: ", style="yellow bold")
            task_content.append(f"{task_name}", style="yellow bold")
            task_content.append(f" (ID: {task_id})", style="yellow dim")
        else:
            task_content.append(f"📋 Task: {task_id}", style="yellow bold")

        task_content.append("\nStatus: ", style="white")
        task_content.append("Executing Task...", style="yellow dim")

        task_branch = None
        if crew_tree:
            task_branch = crew_tree.add(task_content)
            self.print(crew_tree)
        else:
            self.print_panel(task_content, "Task Started", "yellow")

        self.print()

        # Set the current_task_branch attribute directly
        self.current_task_branch = task_branch

        return task_branch

    def update_task_status(
        self,
        crew_tree: Tree | None,
        task_id: str,
        agent_role: str,
        status: str = "completed",
        task_name: str | None = None,
    ) -> None:
        """Update task status in the tree."""
        if not self.verbose or crew_tree is None:
            return

        if status == "completed":
            style = "green"
            status_text = "✅ Completed"
            panel_title = "Task Completion"
        else:
            style = "red"
            status_text = "❌ Failed"
            panel_title = "Task Failure"

        # Update tree label
        for branch in crew_tree.children:
            if str(task_id) in str(branch.label):
                # Build label without introducing stray blank lines
                task_content = Text()
                # First line: Task ID/name
                if task_name:
                    task_content.append("📋 Task: ", style=f"{style} bold")
                    task_content.append(f"{task_name}", style=f"{style} bold")
                    task_content.append(f" (ID: {task_id})", style=f"{style} dim")
                else:
                    task_content.append(f"📋 Task: {task_id}", style=f"{style} bold")

                # Second line: Assigned to
                task_content.append("\nAssigned to: ", style="white")
                task_content.append(agent_role, style=style)

                # Third line: Status
                task_content.append("\nStatus: ", style="white")
                task_content.append(status_text, style=f"{style} bold")
                branch.label = task_content
                self.print(crew_tree)
                break

        # Show status panel
        display_name = task_name if task_name else str(task_id)
        content = self.create_status_content(
            f"Task {status.title()}", display_name, style, Agent=agent_role
        )
        self.print_panel(content, panel_title, style)

    def create_agent_branch(
        self, task_branch: Tree | None, agent_role: str, crew_tree: Tree | None
    ) -> Tree | None:
        """Create and initialize an agent branch."""
        if not self.verbose or not task_branch or not crew_tree:
            return None

        # Instead of creating a separate Agent node, we treat the task branch
        # itself as the logical agent branch so that Reasoning/Tool nodes are
        # nested under the task without an extra visual level.

        # Store the task branch as the current_agent_branch for future nesting.
        self.current_agent_branch = task_branch

        # No additional tree modification needed; return the task branch so
        # caller logic remains unchanged.
        return task_branch

    def update_agent_status(
        self,
        agent_branch: Tree | None,
        agent_role: str,
        crew_tree: Tree | None,
        status: str = "completed",
    ) -> None:
        """Update agent status in the tree."""
        # We no longer render a separate agent branch, so this method simply
        # updates the stored branch reference (already the task branch) without
        # altering the tree. Keeping it a no-op avoids duplicate status lines.
        return

    def create_flow_tree(self, flow_name: str, flow_id: str) -> Tree | None:
        """Create and initialize a flow tree."""
        content = self.create_status_content(
            "Starting Flow Execution", flow_name, "blue", ID=flow_id
        )
        self.print_panel(content, "Flow Execution", "blue", is_flow=True)

        # Create initial tree with flow ID
        flow_label = Text()
        flow_label.append("🌊 Flow: ", style="blue bold")
        flow_label.append(flow_name, style="blue")
        flow_label.append("\nID: ", style="white")
        flow_label.append(flow_id, style="blue")

        flow_tree = Tree(flow_label)
        self.add_tree_node(flow_tree, "✨ Created", "blue")
        self.add_tree_node(flow_tree, "✅ Initialization Complete", "green")

        return flow_tree

    def start_flow(self, flow_name: str, flow_id: str) -> Tree | None:
        """Initialize a flow execution tree."""
        flow_tree = Tree("")
        flow_label = Text()
        flow_label.append("🌊 Flow: ", style="blue bold")
        flow_label.append(flow_name, style="blue")
        flow_label.append("\nID: ", style="white")
        flow_label.append(flow_id, style="blue")
        flow_tree.label = flow_label

        self.add_tree_node(flow_tree, "🧠 Starting Flow...", "yellow")

        self.print(flow_tree)
        self.print()

        self.current_flow_tree = flow_tree
        return flow_tree

    def update_flow_status(
        self,
        flow_tree: Tree | None,
        flow_name: str,
        flow_id: str,
        status: str = "completed",
    ) -> None:
        """Update flow status in the tree."""
        if flow_tree is None:
            return

        # Update main flow label
        self.update_tree_label(
            flow_tree,
            "✅ Flow Finished:" if status == "completed" else "❌ Flow Failed:",
            flow_name,
            "green" if status == "completed" else "red",
        )

        # Update initialization node status
        for child in flow_tree.children:
            if "Starting Flow" in str(child.label):
                child.label = Text(
                    (
                        "✅ Flow Completed"
                        if status == "completed"
                        else "❌ Flow Failed"
                    ),
                    style="green" if status == "completed" else "red",
                )
                break

        content = self.create_status_content(
            (
                "Flow Execution Completed"
                if status == "completed"
                else "Flow Execution Failed"
            ),
            flow_name,
            "green" if status == "completed" else "red",
            ID=flow_id,
        )
        self.print(flow_tree)
        self.print_panel(
            content, "Flow Completion", "green" if status == "completed" else "red"
        )

    def update_method_status(
        self,
        method_branch: Tree | None,
        flow_tree: Tree | None,
        method_name: str,
        status: str = "running",
    ) -> Tree | None:
        """Update method status in the flow tree."""
        if not flow_tree:
            return None

        if status == "running":
            prefix, style = "🔄 Running:", "yellow"
        elif status == "completed":
            prefix, style = "✅ Completed:", "green"
            # Update initialization node when a method completes successfully
            for child in flow_tree.children:
                if "Starting Flow" in str(child.label):
                    child.label = Text("Flow Method Step", style="white")
                    break
        else:
            prefix, style = "❌ Failed:", "red"
            # Update initialization node on failure
            for child in flow_tree.children:
                if "Starting Flow" in str(child.label):
                    child.label = Text("❌ Flow Step Failed", style="red")
                    break

        if not method_branch:
            # Find or create method branch
            for branch in flow_tree.children:
                if method_name in str(branch.label):
                    method_branch = branch
                    break
            if not method_branch:
                method_branch = flow_tree.add("")

        method_branch.label = Text(prefix, style=f"{style} bold") + Text(
            f" {method_name}", style=style
        )

        self.print(flow_tree)
        self.print()
        return method_branch

    def get_llm_tree(self, tool_name: str):
        text = Text()
        text.append(f"🔧 Using {tool_name} from LLM available_function", style="yellow")

        tree = self.current_flow_tree or self.current_crew_tree

        if tree:
            tree.add(text)

        return tree or Tree(text)

    def handle_llm_tool_usage_started(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | str,
    ):
        # Create status content for the tool usage
        content = self.create_status_content(
            "Tool Usage Started", tool_name, Status="In Progress", tool_args=tool_args
        )

        # Create and print the panel
        self.print_panel(content, "Tool Usage", "green")
        self.print()

        # Still return the tree for compatibility with existing code
        return self.get_llm_tree(tool_name)

    def handle_llm_tool_usage_finished(
        self,
        tool_name: str,
    ):
        tree = self.get_llm_tree(tool_name)
        self.add_tree_node(tree, "✅ Tool Usage Completed", "green")
        self.print(tree)
        self.print()

    def handle_llm_tool_usage_error(
        self,
        tool_name: str,
        error: str,
    ):
        tree = self.get_llm_tree(tool_name)
        self.add_tree_node(tree, "❌ Tool Usage Failed", "red")
        self.print(tree)
        self.print()

        error_content = self.create_status_content(
            "Tool Usage Failed", tool_name, "red", Error=error
        )
        self.print_panel(error_content, "Tool Error", "red")

    def handle_tool_usage_started(
        self,
        agent_branch: Tree | None,
        tool_name: str,
        crew_tree: Tree | None,
        tool_args: dict[str, Any] | str = "",
    ) -> Tree | None:
        """Handle tool usage started event."""
        if not self.verbose:
            return None

        # Parent for tool usage: LiteAgent > Agent > Task
        branch_to_use = (
            self.current_lite_agent_branch or agent_branch or self.current_task_branch
        )

        # Render full crew tree when available for consistent live updates
        tree_to_use = self.current_crew_tree or crew_tree or branch_to_use

        if branch_to_use is None or tree_to_use is None:
            # If we don't have a valid branch, default to crew_tree if provided
            if crew_tree is not None:
                branch_to_use = tree_to_use = crew_tree
            else:
                return None

        # Update tool usage count
        self.tool_usage_counts[tool_name] = self.tool_usage_counts.get(tool_name, 0) + 1

        # Find or create tool node
        tool_branch = self.current_tool_branch
        if tool_branch is None:
            tool_branch = branch_to_use.add("")
            self.current_tool_branch = tool_branch

        # Update label with current count
        self.update_tree_label(
            tool_branch,
            "🔧",
            f"Using {tool_name} ({self.tool_usage_counts[tool_name]})",
            "yellow",
        )

        # Print updated tree immediately
        self.print(tree_to_use)
        self.print()

        return tool_branch

    def handle_tool_usage_finished(
        self,
        tool_branch: Tree | None,
        tool_name: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle tool usage finished event."""
        if not self.verbose or tool_branch is None:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch
        if tree_to_use is None:
            return

        # Update the existing tool node's label
        self.update_tree_label(
            tool_branch,
            "🔧",
            f"Used {tool_name} ({self.tool_usage_counts[tool_name]})",
            "green",
        )

        # Clear the current tool branch as we're done with it
        self.current_tool_branch = None

        # Only print if we have a valid tree and the tool node is still in it
        if isinstance(tree_to_use, Tree) and tool_branch in tree_to_use.children:
            self.print(tree_to_use)
            self.print()

    def handle_tool_usage_error(
        self,
        tool_branch: Tree | None,
        tool_name: str,
        error: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle tool usage error event."""
        if not self.verbose:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch

        if tool_branch:
            self.update_tree_label(
                tool_branch,
                "🔧 Failed",
                f"{tool_name} ({self.tool_usage_counts[tool_name]})",
                "red",
            )
            if tree_to_use:
                self.print(tree_to_use)
                self.print()

        # Show error panel
        error_content = self.create_status_content(
            "Tool Usage Failed", tool_name, "red", Error=error
        )
        self.print_panel(error_content, "Tool Error", "red")

    def handle_llm_call_started(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
    ) -> Tree | None:
        """Handle LLM call started event."""
        if not self.verbose:
            return None

        # Parent for tool usage: LiteAgent > Agent > Task
        branch_to_use = (
            self.current_lite_agent_branch or agent_branch or self.current_task_branch
        )

        # Render full crew tree when available for consistent live updates
        tree_to_use = self.current_crew_tree or crew_tree or branch_to_use

        if branch_to_use is None or tree_to_use is None:
            # If we don't have a valid branch, default to crew_tree if provided
            if crew_tree is not None:
                branch_to_use = tree_to_use = crew_tree
            else:
                return None

        # Only add thinking status if we don't have a current tool branch
        # or if the current tool branch is not a thinking node
        should_add_thinking = self.current_tool_branch is None or "Thinking" not in str(
            self.current_tool_branch.label
        )

        if should_add_thinking:
            tool_branch = branch_to_use.add("")
            self.update_tree_label(tool_branch, "🧠", "Thinking...", "blue")
            self.current_tool_branch = tool_branch
            self.print(tree_to_use)
            self.print()
            return tool_branch

        # Return the existing tool branch if it's already a thinking node
        return self.current_tool_branch

    def handle_llm_call_completed(
        self,
        tool_branch: Tree | None,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
    ) -> None:
        """Handle LLM call completed event."""
        if not self.verbose:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch
        if tree_to_use is None:
            return

        # Try to remove the thinking status node - first try the provided tool_branch
        thinking_branch_to_remove = None
        removed = False

        # Method 1: Use the provided tool_branch if it's a thinking node
        if tool_branch is not None and "Thinking" in str(tool_branch.label):
            thinking_branch_to_remove = tool_branch

        # Method 2: Fallback - search for any thinking node if tool_branch is None or not thinking
        if thinking_branch_to_remove is None:
            parents = [
                self.current_lite_agent_branch,
                self.current_agent_branch,
                self.current_task_branch,
                tree_to_use,
            ]
            for parent in parents:
                if isinstance(parent, Tree):
                    for child in parent.children:
                        if "Thinking" in str(child.label):
                            thinking_branch_to_remove = child
                            break
                    if thinking_branch_to_remove:
                        break

        # Remove the thinking node if found
        if thinking_branch_to_remove:
            parents = [
                self.current_lite_agent_branch,
                self.current_agent_branch,
                self.current_task_branch,
                tree_to_use,
            ]
            for parent in parents:
                if (
                    isinstance(parent, Tree)
                    and thinking_branch_to_remove in parent.children
                ):
                    parent.children.remove(thinking_branch_to_remove)
                    removed = True
                    break

            # Clear pointer if we just removed the current_tool_branch
            if self.current_tool_branch is thinking_branch_to_remove:
                self.current_tool_branch = None

            if removed:
                self.print(tree_to_use)
                self.print()

    def handle_llm_call_failed(
        self, tool_branch: Tree | None, error: str, crew_tree: Tree | None
    ) -> None:
        """Handle LLM call failed event."""
        if not self.verbose:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch

        # Find the thinking branch to update (similar to completion logic)
        thinking_branch_to_update = None

        # Method 1: Use the provided tool_branch if it's a thinking node
        if tool_branch is not None and "Thinking" in str(tool_branch.label):
            thinking_branch_to_update = tool_branch

        # Method 2: Fallback - search for any thinking node if tool_branch is None or not thinking
        if thinking_branch_to_update is None:
            parents = [
                self.current_lite_agent_branch,
                self.current_agent_branch,
                self.current_task_branch,
                tree_to_use,
            ]
            for parent in parents:
                if isinstance(parent, Tree):
                    for child in parent.children:
                        if "Thinking" in str(child.label):
                            thinking_branch_to_update = child
                            break
                    if thinking_branch_to_update:
                        break

        # Update the thinking branch to show failure
        if thinking_branch_to_update:
            thinking_branch_to_update.label = Text("❌ LLM Failed", style="red bold")
            # Clear the current_tool_branch reference
            if self.current_tool_branch is thinking_branch_to_update:
                self.current_tool_branch = None
            if tree_to_use:
                self.print(tree_to_use)
                self.print()

        # Show error panel
        error_content = Text()
        error_content.append("❌ LLM Call Failed\n", style="red bold")
        error_content.append("Error: ", style="white")
        error_content.append(str(error), style="red")

        self.print_panel(error_content, "LLM Error", "red")

    def handle_crew_test_started(
        self, crew_name: str, source_id: str, n_iterations: int
    ) -> Tree | None:
        """Handle crew test started event."""
        if not self.verbose:
            return None

        # Create initial panel
        content = Text()
        content.append("🧪 Starting Crew Test\n\n", style="blue bold")
        content.append("Crew: ", style="white")
        content.append(f"{crew_name}\n", style="blue")
        content.append("ID: ", style="white")
        content.append(str(source_id), style="blue")
        content.append("\nIterations: ", style="white")
        content.append(str(n_iterations), style="yellow")

        self.print()
        self.print_panel(content, "Test Execution", "blue")
        self.print()

        # Create and display the test tree
        test_label = Text()
        test_label.append("🧪 Test: ", style="blue bold")
        test_label.append(crew_name or "Crew", style="blue")
        test_label.append("\nStatus: ", style="white")
        test_label.append("In Progress", style="yellow")

        test_tree = Tree(test_label)
        self.add_tree_node(test_tree, "🔄 Running tests...", "yellow")

        self.print(test_tree)
        self.print()
        return test_tree

    def handle_crew_test_completed(
        self, flow_tree: Tree | None, crew_name: str
    ) -> None:
        """Handle crew test completed event."""
        if not self.verbose:
            return

        if flow_tree:
            # Update test tree label to show completion
            test_label = Text()
            test_label.append("✅ Test: ", style="green bold")
            test_label.append(crew_name or "Crew", style="green")
            test_label.append("\nStatus: ", style="white")
            test_label.append("Completed", style="green bold")
            flow_tree.label = test_label

            # Update the running tests node
            for child in flow_tree.children:
                if "Running tests" in str(child.label):
                    child.label = Text("✅ Tests completed successfully", style="green")
                    break

            self.print(flow_tree)
            self.print()

        # Create completion panel
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

    def create_lite_agent_branch(self, lite_agent_role: str) -> Tree | None:
        """Create and initialize a lite agent branch."""
        if not self.verbose:
            return None

        # Create initial tree for LiteAgent if it doesn't exist
        if not self.current_lite_agent_branch:
            lite_agent_label = Text()
            lite_agent_label.append("🤖 LiteAgent: ", style="cyan bold")
            lite_agent_label.append(lite_agent_role, style="cyan")
            lite_agent_label.append("\nStatus: ", style="white")
            lite_agent_label.append("In Progress", style="yellow")

            lite_agent_tree = Tree(lite_agent_label)
            self.current_lite_agent_branch = lite_agent_tree
            self.print(lite_agent_tree)
            self.print()

        return self.current_lite_agent_branch

    def update_lite_agent_status(
        self,
        lite_agent_branch: Tree | None,
        lite_agent_role: str,
        status: str = "completed",
        **fields: dict[str, Any],
    ) -> None:
        """Update lite agent status in the tree."""
        if not self.verbose or lite_agent_branch is None:
            return

        # Determine style based on status
        if status == "completed":
            prefix, style = "✅ LiteAgent:", "green"
            status_text = "Completed"
            title = "LiteAgent Completion"
        elif status == "failed":
            prefix, style = "❌ LiteAgent:", "red"
            status_text = "Failed"
            title = "LiteAgent Error"
        else:
            prefix, style = "🤖 LiteAgent:", "yellow"
            status_text = "In Progress"
            title = "LiteAgent Status"

        # Update the tree label
        lite_agent_label = Text()
        lite_agent_label.append(f"{prefix} ", style=f"{style} bold")
        lite_agent_label.append(lite_agent_role, style=style)
        lite_agent_label.append("\nStatus: ", style="white")
        lite_agent_label.append(status_text, style=f"{style} bold")
        lite_agent_branch.label = lite_agent_label

        self.print(lite_agent_branch)
        self.print()

        # Show status panel if additional fields are provided
        if fields:
            content = self.create_status_content(
                f"LiteAgent {status.title()}", lite_agent_role, style, **fields
            )
            self.print_panel(content, title, style)

    def handle_lite_agent_execution(
        self,
        lite_agent_role: str,
        status: str = "started",
        error: Any = None,
        **fields: dict[str, Any],
    ) -> None:
        """Handle lite agent execution events with consistent formatting."""
        if not self.verbose:
            return

        if status == "started":
            # Create or get the LiteAgent branch
            lite_agent_branch = self.create_lite_agent_branch(lite_agent_role)
            if lite_agent_branch and fields:
                # Show initial status panel
                content = self.create_status_content(
                    "LiteAgent Session Started", lite_agent_role, "cyan", **fields
                )
                self.print_panel(content, "LiteAgent Started", "cyan")
        else:
            # Update existing LiteAgent branch
            if error:
                fields["Error"] = error
            self.update_lite_agent_status(
                self.current_lite_agent_branch, lite_agent_role, status, **fields
            )

    def handle_knowledge_retrieval_started(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
    ) -> Tree | None:
        """Handle knowledge retrieval started event."""
        if not self.verbose:
            return None

        branch_to_use = agent_branch or self.current_lite_agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None or tree_to_use is None:
            # If we don't have a valid branch, default to crew_tree if provided
            if crew_tree is not None:
                branch_to_use = tree_to_use = crew_tree
            else:
                return None

        knowledge_branch = branch_to_use.add("")
        self.update_tree_label(
            knowledge_branch, "🔍", "Knowledge Retrieval Started", "blue"
        )

        self.print(tree_to_use)
        self.print()
        return knowledge_branch

    def handle_knowledge_retrieval_completed(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
        retrieved_knowledge: Any,
    ) -> None:
        """Handle knowledge retrieval completed event."""
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None and tree_to_use is not None:
            branch_to_use = tree_to_use

        if branch_to_use is None or tree_to_use is None:
            if retrieved_knowledge:
                knowledge_text = str(retrieved_knowledge)
                if len(knowledge_text) > 500:
                    knowledge_text = knowledge_text[:497] + "..."

                knowledge_panel = Panel(
                    Text(knowledge_text, style="white"),
                    title="📚 Retrieved Knowledge",
                    border_style="green",
                    padding=(1, 2),
                )
                self.print(knowledge_panel)
                self.print()
            return

        knowledge_branch_found = False
        for child in branch_to_use.children:
            if "Knowledge Retrieval Started" in str(child.label):
                self.update_tree_label(
                    child, "✅", "Knowledge Retrieval Completed", "green"
                )
                knowledge_branch_found = True
                break

        if not knowledge_branch_found:
            for child in branch_to_use.children:
                if (
                    "Knowledge Retrieval" in str(child.label)
                    and "Started" not in str(child.label)
                    and "Completed" not in str(child.label)
                ):
                    self.update_tree_label(
                        child, "✅", "Knowledge Retrieval Completed", "green"
                    )
                    knowledge_branch_found = True
                    break

        if not knowledge_branch_found:
            knowledge_branch = branch_to_use.add("")
            self.update_tree_label(
                knowledge_branch, "✅", "Knowledge Retrieval Completed", "green"
            )

        self.print(tree_to_use)

        if retrieved_knowledge:
            knowledge_text = str(retrieved_knowledge)
            if len(knowledge_text) > 500:
                knowledge_text = knowledge_text[:497] + "..."

            knowledge_panel = Panel(
                Text(knowledge_text, style="white"),
                title="📚 Retrieved Knowledge",
                border_style="green",
                padding=(1, 2),
            )
            self.print(knowledge_panel)

        self.print()

    def handle_knowledge_query_started(
        self,
        agent_branch: Tree | None,
        task_prompt: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle knowledge query generated event."""
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree
        if branch_to_use is None or tree_to_use is None:
            return

        query_branch = branch_to_use.add("")
        self.update_tree_label(
            query_branch, "🔎", f"Query: {task_prompt[:50]}...", "yellow"
        )

        self.print(tree_to_use)
        self.print()

    def handle_knowledge_query_failed(
        self,
        agent_branch: Tree | None,
        error: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle knowledge query failed event."""
        if not self.verbose:
            return

        tree_to_use = self.current_lite_agent_branch or crew_tree
        branch_to_use = self.current_lite_agent_branch or agent_branch

        if branch_to_use and tree_to_use:
            query_branch = branch_to_use.add("")
            self.update_tree_label(query_branch, "❌", "Knowledge Query Failed", "red")
            self.print(tree_to_use)
            self.print()

        # Show error panel
        error_content = self.create_status_content(
            "Knowledge Query Failed", "Query Error", "red", Error=error
        )
        self.print_panel(error_content, "Knowledge Error", "red")

    def handle_knowledge_query_completed(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
    ) -> None:
        """Handle knowledge query completed event."""
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None or tree_to_use is None:
            return

        query_branch = branch_to_use.add("")
        self.update_tree_label(query_branch, "✅", "Knowledge Query Completed", "green")

        self.print(tree_to_use)
        self.print()

    def handle_knowledge_search_query_failed(
        self,
        agent_branch: Tree | None,
        error: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle knowledge search query failed event."""
        if not self.verbose:
            return

        tree_to_use = self.current_lite_agent_branch or crew_tree
        branch_to_use = self.current_lite_agent_branch or agent_branch

        if branch_to_use and tree_to_use:
            query_branch = branch_to_use.add("")
            self.update_tree_label(query_branch, "❌", "Knowledge Search Failed", "red")
            self.print(tree_to_use)
            self.print()

        # Show error panel
        error_content = self.create_status_content(
            "Knowledge Search Failed", "Search Error", "red", Error=error
        )
        self.print_panel(error_content, "Search Error", "red")

    # ----------- AGENT REASONING EVENTS -----------

    def handle_reasoning_started(
        self,
        agent_branch: Tree | None,
        attempt: int,
        crew_tree: Tree | None,
    ) -> Tree | None:
        """Handle agent reasoning started (or refinement) event."""
        if not self.verbose:
            return None

        # Prefer LiteAgent > Agent > Task branch as the parent for reasoning
        branch_to_use = (
            self.current_lite_agent_branch or agent_branch or self.current_task_branch
        )

        # We always want to render the full crew tree when possible so the
        # Live view updates coherently. Fallbacks: crew tree → branch itself.
        tree_to_use = self.current_crew_tree or crew_tree or branch_to_use

        if branch_to_use is None:
            # Nothing to attach to, abort
            return None

        # Reuse existing reasoning branch if present
        reasoning_branch = self.current_reasoning_branch
        if reasoning_branch is None:
            reasoning_branch = branch_to_use.add("")
            self.current_reasoning_branch = reasoning_branch

        # Build label text depending on attempt
        status_text = (
            f"Reasoning (Attempt {attempt})" if attempt > 1 else "Reasoning..."
        )
        self.update_tree_label(reasoning_branch, "🧠", status_text, "blue")

        self.print(tree_to_use)
        self.print()

        return reasoning_branch

    def handle_reasoning_completed(
        self,
        plan: str,
        ready: bool,
        crew_tree: Tree | None,
    ) -> None:
        """Handle agent reasoning completed event."""
        if not self.verbose:
            return

        reasoning_branch = self.current_reasoning_branch
        tree_to_use = (
            self.current_crew_tree
            or self.current_lite_agent_branch
            or self.current_task_branch
            or crew_tree
        )

        style = "green" if ready else "yellow"
        status_text = (
            "Reasoning Completed" if ready else "Reasoning Completed (Not Ready)"
        )

        if reasoning_branch is not None:
            self.update_tree_label(reasoning_branch, "✅", status_text, style)

        if tree_to_use is not None:
            self.print(tree_to_use)

        # Show plan in a panel (trim very long plans)
        if plan:
            plan_panel = Panel(
                Text(plan, style="white"),
                title="🧠 Reasoning Plan",
                border_style=style,
                padding=(1, 2),
            )
            self.print(plan_panel)

        self.print()

        # Clear stored branch after completion
        self.current_reasoning_branch = None

    def handle_reasoning_failed(
        self,
        error: str,
        crew_tree: Tree | None,
    ) -> None:
        """Handle agent reasoning failure event."""
        if not self.verbose:
            return

        reasoning_branch = self.current_reasoning_branch
        tree_to_use = (
            self.current_crew_tree
            or self.current_lite_agent_branch
            or self.current_task_branch
            or crew_tree
        )

        if reasoning_branch is not None:
            self.update_tree_label(reasoning_branch, "❌", "Reasoning Failed", "red")

        if tree_to_use is not None:
            self.print(tree_to_use)

        # Error panel
        error_content = self.create_status_content(
            "Reasoning Failed",
            "Error",
            "red",
            Error=error,
        )
        self.print_panel(error_content, "Reasoning Error", "red")

        # Clear stored branch after failure
        self.current_reasoning_branch = None

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
        import re

        from crewai.agents.parser import AgentAction, AgentFinish

        agent_role = agent_role.partition("\n")[0]

        if isinstance(formatted_answer, AgentAction):
            thought = re.sub(r"\n+", "\n", formatted_answer.thought)
            formatted_json = json.dumps(
                json.loads(formatted_answer.tool_input),
                indent=2,
                ensure_ascii=False,
            )

            # Create content for the action panel
            content = Text()
            content.append("Agent: ", style="white")
            content.append(f"{agent_role}\n\n", style="bright_green bold")

            if thought and thought != "":
                content.append("Thought: ", style="white")
                content.append(f"{thought}\n\n", style="bright_green")

            content.append("Using Tool: ", style="white")
            content.append(f"{formatted_answer.tool}\n\n", style="bright_green bold")

            content.append("Tool Input:\n", style="white")

            # Create a syntax-highlighted JSON code block
            json_syntax = Syntax(
                formatted_json,
                "json",
                theme="monokai",
                line_numbers=False,
                background_color="default",
                word_wrap=True,
            )

            content.append("\n")

            # Create separate panels for better organization
            main_content = Text()
            main_content.append("Agent: ", style="white")
            main_content.append(f"{agent_role}\n\n", style="bright_green bold")

            if thought and thought != "":
                main_content.append("Thought: ", style="white")
                main_content.append(f"{thought}\n\n", style="bright_green")

            main_content.append("Using Tool: ", style="white")
            main_content.append(f"{formatted_answer.tool}", style="bright_green bold")

            # Create the main action panel
            action_panel = Panel(
                main_content,
                title="🔧 Agent Tool Execution",
                border_style="magenta",
                padding=(1, 2),
            )

            # Create the JSON input panel
            input_panel = Panel(
                json_syntax,
                title="Tool Input",
                border_style="blue",
                padding=(1, 2),
            )

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
            self.print(action_panel)
            self.print(input_panel)
            self.print(output_panel)
            self.print()

        elif isinstance(formatted_answer, AgentFinish):
            # Create content for the finish panel
            content = Text()
            content.append("Agent: ", style="white")
            content.append(f"{agent_role}\n\n", style="bright_green bold")
            content.append("Final Answer:\n", style="white")
            content.append(f"{formatted_answer.output}", style="bright_green")

            # Create and display the finish panel
            finish_panel = Panel(
                content,
                title="✅ Agent Final Answer",
                border_style="green",
                padding=(1, 2),
            )
            self.print(finish_panel)
            self.print()

    def handle_memory_retrieval_started(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
    ) -> Tree | None:
        if not self.verbose:
            return None

        branch_to_use = agent_branch or self.current_lite_agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None or tree_to_use is None:
            if crew_tree is not None:
                branch_to_use = tree_to_use = crew_tree
            else:
                return None

        memory_branch = branch_to_use.add("")
        self.update_tree_label(memory_branch, "🧠", "Memory Retrieval Started", "blue")

        self.print(tree_to_use)
        self.print()
        return memory_branch

    def handle_memory_retrieval_completed(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
        memory_content: str,
        retrieval_time_ms: float,
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None and tree_to_use is not None:
            branch_to_use = tree_to_use

        def add_panel():
            memory_text = str(memory_content)
            if len(memory_text) > 500:
                memory_text = memory_text[:497] + "..."

            memory_panel = Panel(
                Text(memory_text, style="white"),
                title="🧠 Retrieved Memory",
                subtitle=f"Retrieval Time: {retrieval_time_ms:.2f}ms",
                border_style="green",
                padding=(1, 2),
            )
            self.print(memory_panel)
            self.print()

        if branch_to_use is None or tree_to_use is None:
            add_panel()
            return

        memory_branch_found = False
        for child in branch_to_use.children:
            if "Memory Retrieval Started" in str(child.label):
                self.update_tree_label(
                    child, "✅", "Memory Retrieval Completed", "green"
                )
                memory_branch_found = True
                break

        if not memory_branch_found:
            for child in branch_to_use.children:
                if (
                    "Memory Retrieval" in str(child.label)
                    and "Started" not in str(child.label)
                    and "Completed" not in str(child.label)
                ):
                    self.update_tree_label(
                        child, "✅", "Memory Retrieval Completed", "green"
                    )
                    memory_branch_found = True
                    break

        if not memory_branch_found:
            memory_branch = branch_to_use.add("")
            self.update_tree_label(
                memory_branch, "✅", "Memory Retrieval Completed", "green"
            )

        self.print(tree_to_use)

        if memory_content:
            add_panel()

    def handle_memory_query_completed(
        self,
        agent_branch: Tree | None,
        source_type: str,
        query_time_ms: float,
        crew_tree: Tree | None,
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None and tree_to_use is not None:
            branch_to_use = tree_to_use

        if branch_to_use is None:
            return

        memory_type = source_type.replace("_", " ").title()

        for child in branch_to_use.children:
            if "Memory Retrieval" in str(child.label):
                for inner_child in child.children:
                    sources_branch = inner_child
                    if "Sources Used" in str(inner_child.label):
                        sources_branch.add(f"✅ {memory_type} ({query_time_ms:.2f}ms)")
                        break
                else:
                    sources_branch = child.add("Sources Used")
                    sources_branch.add(f"✅ {memory_type} ({query_time_ms:.2f}ms)")
                    break

    def handle_memory_query_failed(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
        error: str,
        source_type: str,
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None and tree_to_use is not None:
            branch_to_use = tree_to_use

        if branch_to_use is None:
            return

        memory_type = source_type.replace("_", " ").title()

        for child in branch_to_use.children:
            if "Memory Retrieval" in str(child.label):
                for inner_child in child.children:
                    sources_branch = inner_child
                    if "Sources Used" in str(inner_child.label):
                        sources_branch.add(f"❌ {memory_type} - Error: {error}")
                        break
                else:
                    sources_branch = child.add("🧠 Sources Used")
                    sources_branch.add(f"❌ {memory_type} - Error: {error}")
                    break

    def handle_memory_save_started(
        self, agent_branch: Tree | None, crew_tree: Tree | None
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = agent_branch or self.current_lite_agent_branch
        tree_to_use = branch_to_use or crew_tree

        if tree_to_use is None:
            return

        for child in tree_to_use.children:
            if "Memory Update" in str(child.label):
                break
        else:
            memory_branch = tree_to_use.add("")
            self.update_tree_label(
                memory_branch, "🧠", "Memory Update Overall", "white"
            )

        self.print(tree_to_use)
        self.print()

    def handle_memory_save_completed(
        self,
        agent_branch: Tree | None,
        crew_tree: Tree | None,
        save_time_ms: float,
        source_type: str,
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = agent_branch or self.current_lite_agent_branch
        tree_to_use = branch_to_use or crew_tree

        if tree_to_use is None:
            return

        memory_type = source_type.replace("_", " ").title()
        content = f"✅ {memory_type} Memory Saved ({save_time_ms:.2f}ms)"

        for child in tree_to_use.children:
            if "Memory Update" in str(child.label):
                child.add(content)
                break
        else:
            memory_branch = tree_to_use.add("")
            memory_branch.add(content)

        self.print(tree_to_use)
        self.print()

    def handle_memory_save_failed(
        self,
        agent_branch: Tree | None,
        error: str,
        source_type: str,
        crew_tree: Tree | None,
    ) -> None:
        if not self.verbose:
            return

        branch_to_use = agent_branch or self.current_lite_agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None or tree_to_use is None:
            return

        memory_type = source_type.replace("_", " ").title()
        content = f"❌ {memory_type} Memory Save Failed"
        for child in branch_to_use.children:
            if "Memory Update" in str(child.label):
                child.add(content)
                break
        else:
            memory_branch = branch_to_use.add("")
            memory_branch.add(content)

        self.print(tree_to_use)
        self.print()

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

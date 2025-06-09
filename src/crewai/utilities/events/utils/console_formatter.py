from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.live import Live


class ConsoleFormatter:
    current_crew_tree: Optional[Tree] = None
    current_task_branch: Optional[Tree] = None
    current_agent_branch: Optional[Tree] = None
    current_tool_branch: Optional[Tree] = None
    current_flow_tree: Optional[Tree] = None
    current_method_branch: Optional[Tree] = None
    current_lite_agent_branch: Optional[Tree] = None
    tool_usage_counts: Dict[str, int] = {}
    current_reasoning_branch: Optional[Tree] = None  # Track reasoning status
    current_llm_tool_tree: Optional[Tree] = None

    def __init__(self, verbose: bool = False):
        self.console = Console(width=None)
        self.verbose = verbose
        # Live instance to dynamically update a Tree renderable (e.g. the Crew tree)
        # When multiple Tree objects are printed sequentially we reuse this Live
        # instance so the previous render is replaced instead of writing a new one.
        # Once any non-Tree renderable is printed we stop the Live session so the
        # final Tree persists on the terminal.
        self._live: Optional[Live] = None

    def create_panel(self, content: Text, title: str, style: str = "blue") -> Panel:
        """Create a standardized panel with consistent styling."""
        return Panel(
            content,
            title=title,
            border_style=style,
            padding=(1, 2),
        )

    def create_status_content(
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

    def update_tree_label(
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

        # Case 2: blank line while a live session is running – ignore so we
        # don't break the in-place rendering behaviour
        if len(args) == 0 and self._live:
            return

        # Case 3: printing something other than a Tree → terminate live session
        if self._live:
            self._live.stop()
            self._live = None

        # Finally, pass through to the regular Console.print implementation
        self.console.print(*args, **kwargs)

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
        tree: Optional[Tree],
        crew_name: str,
        source_id: str,
        status: str = "completed",
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

        self.print_panel(content, title, style)

    def create_crew_tree(self, crew_name: str, source_id: str) -> Optional[Tree]:
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
        self, crew_tree: Optional[Tree], task_id: str
    ) -> Optional[Tree]:
        """Create and initialize a task branch."""
        if not self.verbose:
            return None

        task_content = Text()
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
        crew_tree: Optional[Tree],
        task_id: str,
        agent_role: str,
        status: str = "completed",
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
                # First line: Task ID
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
        content = self.create_status_content(
            f"Task {status.title()}", str(task_id), style, Agent=agent_role
        )
        self.print_panel(content, panel_title, style)

    def create_agent_branch(
        self, task_branch: Optional[Tree], agent_role: str, crew_tree: Optional[Tree]
    ) -> Optional[Tree]:
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
        agent_branch: Optional[Tree],
        agent_role: str,
        crew_tree: Optional[Tree],
        status: str = "completed",
    ) -> None:
        """Update agent status in the tree."""
        # We no longer render a separate agent branch, so this method simply
        # updates the stored branch reference (already the task branch) without
        # altering the tree. Keeping it a no-op avoids duplicate status lines.
        return

    def create_flow_tree(self, flow_name: str, flow_id: str) -> Optional[Tree]:
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

    def start_flow(self, flow_name: str, flow_id: str) -> Optional[Tree]:
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
        flow_tree: Optional[Tree],
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
        method_branch: Optional[Tree],
        flow_tree: Optional[Tree],
        method_name: str,
        status: str = "running",
    ) -> Optional[Tree]:
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
    ):
        tree = self.get_llm_tree(tool_name)
        self.add_tree_node(tree, "🔄 Tool Usage Started", "green")
        self.print(tree)
        self.print()
        return tree

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
        agent_branch: Optional[Tree],
        tool_name: str,
        crew_tree: Optional[Tree],
    ) -> Optional[Tree]:
        """Handle tool usage started event."""
        if not self.verbose:
            return None

        # Parent for tool usage: LiteAgent > Agent > Task
        branch_to_use = (
            self.current_lite_agent_branch
            or agent_branch
            or self.current_task_branch
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
        tool_branch: Optional[Tree],
        tool_name: str,
        crew_tree: Optional[Tree],
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
        tool_branch: Optional[Tree],
        tool_name: str,
        error: str,
        crew_tree: Optional[Tree],
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
        agent_branch: Optional[Tree],
        crew_tree: Optional[Tree],
    ) -> Optional[Tree]:
        """Handle LLM call started event."""
        if not self.verbose:
            return None

        # Parent for tool usage: LiteAgent > Agent > Task
        branch_to_use = (
            self.current_lite_agent_branch
            or agent_branch
            or self.current_task_branch
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
        if self.current_tool_branch is None:
            tool_branch = branch_to_use.add("")
            self.update_tree_label(tool_branch, "🧠", "Thinking...", "blue")
            self.current_tool_branch = tool_branch
            self.print(tree_to_use)
            self.print()
            return tool_branch
        return None

    def handle_llm_call_completed(
        self,
        tool_branch: Optional[Tree],
        agent_branch: Optional[Tree],
        crew_tree: Optional[Tree],
    ) -> None:
        """Handle LLM call completed event."""
        if not self.verbose or tool_branch is None:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch
        if tree_to_use is None:
            return

        # Remove the thinking status node when complete
        if "Thinking" in str(tool_branch.label):
            parents = [
                self.current_lite_agent_branch,
                self.current_agent_branch,
                self.current_task_branch,
                tree_to_use,
            ]
            removed = False
            for parent in parents:
                if isinstance(parent, Tree) and tool_branch in parent.children:
                    parent.children.remove(tool_branch)
                    removed = True
                    break

            # Clear pointer if we just removed the current_tool_branch
            if self.current_tool_branch is tool_branch:
                self.current_tool_branch = None

            if removed:
                self.print(tree_to_use)
                self.print()

    def handle_llm_call_failed(
        self, tool_branch: Optional[Tree], error: str, crew_tree: Optional[Tree]
    ) -> None:
        """Handle LLM call failed event."""
        if not self.verbose:
            return

        # Decide which tree to render: prefer full crew tree, else parent branch
        tree_to_use = self.current_crew_tree or crew_tree or self.current_task_branch

        # Update tool branch if it exists
        if tool_branch:
            tool_branch.label = Text("❌ LLM Failed", style="red bold")
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
    ) -> Optional[Tree]:
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
        self, flow_tree: Optional[Tree], crew_name: str
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

    def create_lite_agent_branch(self, lite_agent_role: str) -> Optional[Tree]:
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
        lite_agent_branch: Optional[Tree],
        lite_agent_role: str,
        status: str = "completed",
        **fields: Dict[str, Any],
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
        **fields: Dict[str, Any],
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
        agent_branch: Optional[Tree],
        crew_tree: Optional[Tree],
    ) -> Optional[Tree]:
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
        agent_branch: Optional[Tree],
        crew_tree: Optional[Tree],
        retrieved_knowledge: Any,
    ) -> None:
        """Handle knowledge retrieval completed event."""
        if not self.verbose:
            return None

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
            return None

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
        agent_branch: Optional[Tree],
        task_prompt: str,
        crew_tree: Optional[Tree],
    ) -> None:
        """Handle knowledge query generated event."""
        if not self.verbose:
            return None

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree
        if branch_to_use is None or tree_to_use is None:
            return None

        query_branch = branch_to_use.add("")
        self.update_tree_label(
            query_branch, "🔎", f"Query: {task_prompt[:50]}...", "yellow"
        )

        self.print(tree_to_use)
        self.print()

    def handle_knowledge_query_failed(
        self,
        agent_branch: Optional[Tree],
        error: str,
        crew_tree: Optional[Tree],
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
        agent_branch: Optional[Tree],
        crew_tree: Optional[Tree],
    ) -> None:
        """Handle knowledge query completed event."""
        if not self.verbose:
            return None

        branch_to_use = self.current_lite_agent_branch or agent_branch
        tree_to_use = branch_to_use or crew_tree

        if branch_to_use is None or tree_to_use is None:
            return None

        query_branch = branch_to_use.add("")
        self.update_tree_label(query_branch, "✅", "Knowledge Query Completed", "green")

        self.print(tree_to_use)
        self.print()

    def handle_knowledge_search_query_failed(
        self,
        agent_branch: Optional[Tree],
        error: str,
        crew_tree: Optional[Tree],
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
        agent_branch: Optional[Tree],
        attempt: int,
        crew_tree: Optional[Tree],
    ) -> Optional[Tree]:
        """Handle agent reasoning started (or refinement) event."""
        if not self.verbose:
            return None

        # Prefer LiteAgent > Agent > Task branch as the parent for reasoning
        branch_to_use = (
            self.current_lite_agent_branch
            or agent_branch
            or self.current_task_branch
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
        crew_tree: Optional[Tree],
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
        status_text = "Reasoning Completed" if ready else "Reasoning Completed (Not Ready)"

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
        crew_tree: Optional[Tree],
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

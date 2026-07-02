from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks import (
    clear_all_tool_call_hooks,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
)
import pytest


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "Test tool description"
    return tool


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.role = "Test Agent"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = Mock()
    task.description = "Test task"
    return task


@pytest.fixture
def mock_crew():
    """Create a mock crew for testing."""
    crew = Mock()
    return crew


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    from crewai.hooks import tool_hooks

    original_before = tool_hooks._before_tool_call_hooks.copy()
    original_after = tool_hooks._after_tool_call_hooks.copy()

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()

    yield

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.extend(original_before)
    tool_hooks._after_tool_call_hooks.extend(original_after)


class TestToolCallHookContext:
    """Test ToolCallHookContext initialization and attributes."""

    def test_context_initialization(self, mock_tool, mock_agent, mock_task, mock_crew):
        """Test that context is initialized correctly."""
        tool_input = {"arg1": "value1", "arg2": "value2"}

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            agent=mock_agent,
            task=mock_task,
            crew=mock_crew,
        )

        assert context.tool_name == "test_tool"
        assert context.tool_input == tool_input
        assert context.tool == mock_tool
        assert context.agent == mock_agent
        assert context.task == mock_task
        assert context.crew == mock_crew
        assert context.tool_result is None
        assert context.raw_tool_result is None

    def test_context_with_result(self, mock_tool):
        """Test that context includes result when provided."""
        tool_input = {"arg1": "value1"}
        tool_result = "Test tool result"
        raw_tool_result = {"value": 42}

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=tool_result,
            raw_tool_result=raw_tool_result,
        )

        assert context.tool_result == tool_result
        assert context.raw_tool_result == raw_tool_result

    def test_tool_input_is_mutable_reference(self, mock_tool):
        """Test that modifying context.tool_input modifies the original dict."""
        tool_input = {"arg1": "value1"}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        context.tool_input["arg2"] = "value2"

        assert "arg2" in tool_input
        assert tool_input["arg2"] == "value2"


class TestBeforeToolCallHooks:
    """Test before_tool_call hook registration and execution."""

    def test_register_before_hook(self):
        """Test that before hooks are registered correctly."""
        def test_hook(context):
            return None

        register_before_tool_call_hook(test_hook)
        hooks = get_before_tool_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_before_hooks(self):
        """Test that multiple before hooks can be registered."""
        def hook1(context):
            return None

        def hook2(context):
            return None

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        hooks = get_before_tool_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_before_hook_can_block_execution(self, mock_tool):
        """Test that before hooks can block tool execution."""
        def block_hook(context):
            if context.tool_name == "dangerous_tool":
                return False
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="dangerous_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        result = block_hook(context)
        assert result is False

    def test_before_hook_can_allow_execution(self, mock_tool):
        """Test that before hooks can explicitly allow execution."""
        def allow_hook(context):
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="safe_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        result = allow_hook(context)
        assert result is None

    def test_before_hook_can_modify_input(self, mock_tool):
        """Test that before hooks can modify tool input in-place."""
        def modify_input_hook(context):
            context.tool_input["modified_by_hook"] = True
            return

        tool_input = {"arg1": "value1"}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        modify_input_hook(context)

        assert "modified_by_hook" in context.tool_input
        assert context.tool_input["modified_by_hook"] is True

    def test_get_before_hooks_returns_copy(self):
        """Test that get_before_tool_call_hooks returns a copy."""
        def test_hook(context):
            return None

        register_before_tool_call_hook(test_hook)
        hooks1 = get_before_tool_call_hooks()
        hooks2 = get_before_tool_call_hooks()

        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestAfterToolCallHooks:
    """Test after_tool_call hook registration and execution."""

    def test_register_after_hook(self):
        """Test that after hooks are registered correctly."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        hooks = get_after_tool_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_after_hooks(self):
        """Test that multiple after hooks can be registered."""
        def hook1(context):
            return None

        def hook2(context):
            return None

        register_after_tool_call_hook(hook1)
        register_after_tool_call_hook(hook2)
        hooks = get_after_tool_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_after_hook_can_modify_result(self, mock_tool):
        """Test that after hooks can modify the tool result."""
        original_result = "Original result"

        def modify_result_hook(context):
            if context.tool_result:
                return context.tool_result.replace("Original", "Modified")
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=original_result,
        )

        modified = modify_result_hook(context)
        assert modified == "Modified result"

    def test_after_hook_returns_none_keeps_original(self, mock_tool):
        """Test that returning None keeps the original result."""
        original_result = "Original result"

        def no_change_hook(context):
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=original_result,
        )

        result = no_change_hook(context)

        assert result is None
        assert context.tool_result == original_result

    def test_get_after_hooks_returns_copy(self):
        """Test that get_after_tool_call_hooks returns a copy."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        hooks1 = get_after_tool_call_hooks()
        hooks2 = get_after_tool_call_hooks()

        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestToolHooksIntegration:
    """Test integration scenarios with multiple hooks."""

    def test_multiple_before_hooks_execute_in_order(self, mock_tool):
        """Test that multiple before hooks execute in registration order."""
        execution_order = []

        def hook1(context):
            execution_order.append(1)
            return

        def hook2(context):
            execution_order.append(2)
            return

        def hook3(context):
            execution_order.append(3)
            return

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        register_before_tool_call_hook(hook3)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        for hook in hooks:
            hook(context)

        assert execution_order == [1, 2, 3]

    def test_first_blocking_hook_stops_execution(self, mock_tool):
        """Test that first hook returning False blocks execution."""
        execution_order = []

        def hook1(context):
            execution_order.append(1)
            return

        def hook2(context):
            execution_order.append(2)
            return False

        def hook3(context):
            execution_order.append(3)
            return

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        register_before_tool_call_hook(hook3)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        blocked = False
        for hook in hooks:
            result = hook(context)
            if result is False:
                blocked = True
                break

        assert blocked is True
        assert execution_order == [1, 2]

    def test_multiple_after_hooks_chain_modifications(self, mock_tool):
        """Test that multiple after hooks can chain modifications."""
        def hook1(context):
            if context.tool_result:
                return context.tool_result + " [hook1]"
            return None

        def hook2(context):
            if context.tool_result:
                return context.tool_result + " [hook2]"
            return None

        register_after_tool_call_hook(hook1)
        register_after_tool_call_hook(hook2)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result="Original",
        )

        hooks = get_after_tool_call_hooks()

        result = context.tool_result
        for hook in hooks:
            context.tool_result = result
            modified = hook(context)
            if modified is not None:
                result = modified

        assert result == "Original [hook1] [hook2]"

    def test_hooks_with_validation_and_sanitization(self, mock_tool):
        """Test a realistic scenario with validation and sanitization hooks."""
        # Validation hook (before)
        def validate_file_path(context):
            if context.tool_name == "write_file":
                file_path = context.tool_input.get("file_path", "")
                if ".env" in file_path:
                    return False
            return None

        # Sanitization hook (after)
        def sanitize_secrets(context):
            if context.tool_result and "SECRET_KEY" in context.tool_result:
                return context.tool_result.replace("SECRET_KEY=abc123", "SECRET_KEY=[REDACTED]")
            return None

        register_before_tool_call_hook(validate_file_path)
        register_after_tool_call_hook(sanitize_secrets)

        blocked_context = ToolCallHookContext(
            tool_name="write_file",
            tool_input={"file_path": ".env"},
            tool=mock_tool,
        )

        before_hooks = get_before_tool_call_hooks()
        blocked = False
        for hook in before_hooks:
            if hook(blocked_context) is False:
                blocked = True
                break

        assert blocked is True

        sanitize_context = ToolCallHookContext(
            tool_name="read_file",
            tool_input={"file_path": "config.txt"},
            tool=mock_tool,
            tool_result="Content: SECRET_KEY=abc123",
        )

        after_hooks = get_after_tool_call_hooks()
        result = sanitize_context.tool_result
        for hook in after_hooks:
            sanitize_context.tool_result = result
            modified = hook(sanitize_context)
            if modified is not None:
                result = modified

        assert "SECRET_KEY=[REDACTED]" in result
        assert "abc123" not in result


    def test_unregister_before_hook(self):
        """Test that before hooks can be unregistered."""
        def test_hook(context):
            pass

        register_before_tool_call_hook(test_hook)
        unregister_before_tool_call_hook(test_hook)
        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 0

    def test_unregister_after_hook(self):
        """Test that after hooks can be unregistered."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        unregister_after_tool_call_hook(test_hook)
        hooks = get_after_tool_call_hooks()
        assert len(hooks) == 0

    def test_clear_all_tool_call_hooks(self):
        """Test that all tool call hooks can be cleared."""
        def test_hook(context):
            pass

        register_before_tool_call_hook(test_hook)
        register_after_tool_call_hook(test_hook)
        clear_all_tool_call_hooks()
        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 0

    @pytest.mark.vcr()
    def test_lite_agent_hooks_integration_with_real_tool(self):
        """Test that LiteAgent executes before/after tool call hooks with real tool calls."""
        import os

        from crewai.lite_agent import LiteAgent
        from crewai.tools import tool

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real tool test")

        hook_calls = {"before": [], "after": []}

        @tool("calculate_sum")
        def calculate_sum(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def before_tool_call_hook(context: ToolCallHookContext) -> bool:
            """Log and verify before hook execution."""
            print(f"\n[BEFORE HOOK] Tool: {context.tool_name}")
            print(f"[BEFORE HOOK] Tool input: {context.tool_input}")
            print(f"[BEFORE HOOK] Agent: {context.agent.role if context.agent else 'None'}")
            print(f"[BEFORE HOOK] Task: {context.task}")
            print(f"[BEFORE HOOK] Crew: {context.crew}")

            hook_calls["before"].append({
                "tool_name": context.tool_name,
                "tool_input": context.tool_input,
                "has_agent": context.agent is not None,
                "has_task": context.task is not None,
                "has_crew": context.crew is not None,
            })

            return True

        def after_tool_call_hook(context: ToolCallHookContext) -> str | None:
            """Log and verify after hook execution."""
            print(f"\n[AFTER HOOK] Tool: {context.tool_name}")
            print(f"[AFTER HOOK] Tool result: {context.tool_result}")
            print(f"[AFTER HOOK] Agent: {context.agent.role if context.agent else 'None'}")

            hook_calls["after"].append({
                "tool_name": context.tool_name,
                "tool_result": context.tool_result,
                "has_result": context.tool_result is not None,
            })

            return None

        register_before_tool_call_hook(before_tool_call_hook)
        register_after_tool_call_hook(after_tool_call_hook)

        try:
            lite_agent = LiteAgent(
                role="Calculator Assistant",
                goal="Help with math calculations",
                backstory="You are a helpful calculator assistant",
                tools=[calculate_sum],
                verbose=True,
            )

            result = lite_agent.kickoff("What is 5 + 3? Use the calculate_sum tool.")

            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            assert len(hook_calls["after"]) > 0, "After hook was never called"

            # LiteAgent doesn't have task/crew context, unlike agents in CrewBase
            before_call = hook_calls["before"][0]
            assert before_call["tool_name"] == "calculate_sum", "Tool name should be 'calculate_sum'"
            assert "a" in before_call["tool_input"], "Tool input should have 'a' parameter"
            assert "b" in before_call["tool_input"], "Tool input should have 'b' parameter"

            after_call = hook_calls["after"][0]
            assert after_call["has_result"] is True, "After hook should have tool result"
            assert after_call["tool_name"] == "calculate_sum", "Tool name should match"
            assert "8" in str(after_call["tool_result"]), "Tool result should contain the sum"

        finally:
            unregister_before_tool_call_hook(before_tool_call_hook)
            unregister_after_tool_call_hook(after_tool_call_hook)


class TestNativeToolCallingHooksIntegration:
    """Integration tests for hooks with native function calling (Agent and Crew)."""

    @pytest.mark.vcr()
    def test_agent_native_tool_hooks_before_and_after(self):
        """Test that Agent with native tool calling executes before/after hooks."""
        from crewai import Agent
        from crewai.tools import tool

        hook_calls = {"before": [], "after": []}

        @tool("multiply_numbers")
        def multiply_numbers(a: int, b: int) -> int:
            """Multiply two numbers together."""
            return a * b

        def before_hook(context: ToolCallHookContext) -> bool | None:
            hook_calls["before"].append({
                "tool_name": context.tool_name,
                "tool_input": dict(context.tool_input),
                "has_agent": context.agent is not None,
            })
            return None

        def after_hook(context: ToolCallHookContext) -> str | None:
            hook_calls["after"].append({
                "tool_name": context.tool_name,
                "tool_result": context.tool_result,
                "has_agent": context.agent is not None,
            })
            return None

        register_before_tool_call_hook(before_hook)
        register_after_tool_call_hook(after_hook)

        try:
            agent = Agent(
                role="Calculator",
                goal="Perform calculations",
                backstory="You are a calculator assistant",
                tools=[multiply_numbers],
                verbose=True,
            )

            agent.kickoff(
                messages="What is 7 times 6? Use the multiply_numbers tool."
            )

            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            before_call = hook_calls["before"][0]
            assert before_call["tool_name"] == "multiply_numbers"
            assert "a" in before_call["tool_input"]
            assert "b" in before_call["tool_input"]
            assert before_call["has_agent"] is True

            assert len(hook_calls["after"]) > 0, "After hook was never called"
            after_call = hook_calls["after"][0]
            assert after_call["tool_name"] == "multiply_numbers"
            assert "42" in str(after_call["tool_result"])
            assert after_call["has_agent"] is True

        finally:
            unregister_before_tool_call_hook(before_hook)
            unregister_after_tool_call_hook(after_hook)

    @pytest.mark.vcr()
    def test_crew_native_tool_hooks_before_and_after(self):
        """Test that Crew with Agent executes before/after hooks with full context."""
        from crewai import Agent, Crew, Task
        from crewai.tools import tool


        hook_calls = {"before": [], "after": []}

        @tool("divide_numbers")
        def divide_numbers(a: int, b: int) -> float:
            """Divide first number by second number."""
            return a / b

        def before_hook(context: ToolCallHookContext) -> bool | None:
            hook_calls["before"].append({
                "tool_name": context.tool_name,
                "tool_input": dict(context.tool_input),
                "has_agent": context.agent is not None,
                "has_task": context.task is not None,
                "has_crew": context.crew is not None,
                "agent_role": context.agent.role if context.agent else None,
            })
            return None

        def after_hook(context: ToolCallHookContext) -> str | None:
            hook_calls["after"].append({
                "tool_name": context.tool_name,
                "tool_result": context.tool_result,
                "has_agent": context.agent is not None,
                "has_task": context.task is not None,
                "has_crew": context.crew is not None,
            })
            return None

        register_before_tool_call_hook(before_hook)
        register_after_tool_call_hook(after_hook)

        try:
            agent = Agent(
                role="Math Assistant",
                goal="Perform division calculations accurately",
                backstory="You are a math assistant that helps with division",
                tools=[divide_numbers],
                verbose=True,
            )

            task = Task(
                description="Calculate 100 divided by 4 using the divide_numbers tool.",
                expected_output="The result of the division",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
            )

            crew.kickoff()

            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            before_call = hook_calls["before"][0]
            assert before_call["tool_name"] == "divide_numbers"
            assert "a" in before_call["tool_input"]
            assert "b" in before_call["tool_input"]
            assert before_call["has_agent"] is True
            assert before_call["has_task"] is True
            assert before_call["has_crew"] is True
            assert before_call["agent_role"] == "Math Assistant"

            assert len(hook_calls["after"]) > 0, "After hook was never called"
            after_call = hook_calls["after"][0]
            assert after_call["tool_name"] == "divide_numbers"
            assert "25" in str(after_call["tool_result"])
            assert after_call["has_agent"] is True
            assert after_call["has_task"] is True
            assert after_call["has_crew"] is True

        finally:
            unregister_before_tool_call_hook(before_hook)
            unregister_after_tool_call_hook(after_hook)

    @pytest.mark.vcr()
    def test_before_hook_blocks_tool_execution_in_crew(self):
        """Test that returning False from before hook blocks tool execution."""
        from crewai import Agent, Crew, Task
        from crewai.tools import tool

        hook_calls = {"before": [], "after": [], "tool_executed": False}

        @tool("dangerous_operation")
        def dangerous_operation(action: str) -> str:
            """Perform a dangerous operation that should be blocked."""
            hook_calls["tool_executed"] = True
            return f"Executed: {action}"

        def blocking_before_hook(context: ToolCallHookContext) -> bool | None:
            hook_calls["before"].append({
                "tool_name": context.tool_name,
                "tool_input": dict(context.tool_input),
            })
            # Block all calls to dangerous_operation
            if context.tool_name == "dangerous_operation":
                return False
            return None

        def after_hook(context: ToolCallHookContext) -> str | None:
            hook_calls["after"].append({
                "tool_name": context.tool_name,
                "tool_result": context.tool_result,
            })
            return None

        register_before_tool_call_hook(blocking_before_hook)
        register_after_tool_call_hook(after_hook)

        try:
            agent = Agent(
                role="Test Agent",
                goal="Try to use the dangerous operation tool",
                backstory="You are a test agent",
                tools=[dangerous_operation],
                verbose=True,
            )

            task = Task(
                description="Use the dangerous_operation tool with action 'delete_all'.",
                expected_output="The result of the operation",
                agent=agent,
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
            )

            crew.kickoff()

            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            before_call = hook_calls["before"][0]
            assert before_call["tool_name"] == "dangerous_operation"

            assert hook_calls["tool_executed"] is False, "Tool should have been blocked"

            assert len(hook_calls["after"]) > 0, "After hook was never called"
            after_call = hook_calls["after"][0]
            assert "blocked" in after_call["tool_result"].lower()

        finally:
            unregister_before_tool_call_hook(blocking_before_hook)
            unregister_after_tool_call_hook(after_hook)


class TestCrewLevelToolCallHooks:
    """Tests for crew-level before_tool_call and after_tool_call hooks."""

    def test_crew_before_tool_call_blocks_execution(self):
        """Test that crew.before_tool_call blocks tool execution when it raises."""
        from crewai.agents.parser import AgentAction
        from crewai.tools.tool_types import ToolResult
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "restricted_tool"
        mock_tool.description = "A restricted tool"
        mock_tool.result_as_answer = False
        mock_tool.args_schema = None

        mock_agent = Mock()
        mock_agent.role = "Researcher"

        mock_crew = Mock()

        def before_tool_call(agent, tool_name, tool_input):
            if tool_name == "restricted_tool":
                raise PermissionError("restricted_tool requires Admin role")

        mock_crew.before_tool_call = before_tool_call
        mock_crew.after_tool_call = None
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: restricted_tool\nAction Input: {}",
            thought="I should use restricted_tool",
            tool="restricted_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="restricted_tool", arguments={}
                ),
            )

            result = execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert isinstance(result, ToolResult)
        assert "restricted_tool requires Admin role" in result.result
        assert result.result_as_answer is False

    def test_crew_before_tool_call_allows_execution(self):
        """Test that crew.before_tool_call allows execution when it doesn't raise."""
        from crewai.agents.parser import AgentAction
        from crewai.tools.tool_types import ToolResult
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "allowed_tool"
        mock_tool.description = "An allowed tool"
        mock_tool.result_as_answer = False
        mock_tool.args_schema = None

        mock_agent = Mock()
        mock_agent.role = "Admin"

        call_log = []

        def before_tool_call(agent, tool_name, tool_input):
            call_log.append({
                "agent_role": agent.role,
                "tool_name": tool_name,
                "tool_input": tool_input,
            })

        mock_crew = Mock()
        mock_crew.before_tool_call = before_tool_call
        mock_crew.after_tool_call = None
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: allowed_tool\nAction Input: {}",
            thought="Use the tool",
            tool="allowed_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="allowed_tool", arguments={"query": "test"}
                ),
            )
            mp.setattr(
                ToolUsage,
                "use",
                lambda self, calling, text: "Tool result",
            )

            result = execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert len(call_log) == 1
        assert call_log[0]["agent_role"] == "Admin"
        assert call_log[0]["tool_name"] == "allowed_tool"
        assert result.result == "Tool result"

    def test_crew_after_tool_call_receives_output(self):
        """Test that crew.after_tool_call receives the correct tool output."""
        from crewai.agents.parser import AgentAction
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "my_tool"
        mock_tool.description = "A tool"
        mock_tool.result_as_answer = False
        mock_tool.args_schema = None

        mock_agent = Mock()
        mock_agent.role = "Researcher"

        after_log = []

        def after_tool_call(agent, tool_name, tool_input, tool_output):
            after_log.append({
                "agent_role": agent.role,
                "tool_name": tool_name,
                "tool_output": tool_output,
            })

        mock_crew = Mock()
        mock_crew.before_tool_call = None
        mock_crew.after_tool_call = after_tool_call
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: my_tool\nAction Input: {}",
            thought="Use the tool",
            tool="my_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="my_tool", arguments={"query": "AI"}
                ),
            )
            mp.setattr(
                ToolUsage,
                "use",
                lambda self, calling, text: "Result for: AI",
            )

            execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert len(after_log) == 1
        assert after_log[0]["tool_name"] == "my_tool"
        assert after_log[0]["tool_output"] == "Result for: AI"
        assert after_log[0]["agent_role"] == "Researcher"

    def test_crew_before_blocks_prevents_after_call(self):
        """Test that when before_tool_call blocks, after_tool_call is not called."""
        from crewai.agents.parser import AgentAction
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "my_tool"
        mock_tool.description = "A tool"
        mock_tool.result_as_answer = False

        mock_agent = Mock()
        mock_agent.role = "Researcher"

        after_mock = Mock()

        def before_tool_call(agent, tool_name, tool_input):
            raise PermissionError("Blocked!")

        mock_crew = Mock()
        mock_crew.before_tool_call = before_tool_call
        mock_crew.after_tool_call = after_mock
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: my_tool\nAction Input: {}",
            thought="Use the tool",
            tool="my_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="my_tool", arguments={}
                ),
            )

            result = execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert "Blocked!" in result.result
        after_mock.assert_not_called()

    def test_crew_without_hooks_works_normally(self):
        """Test that crews without hooks work normally."""
        from crewai.agents.parser import AgentAction
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "my_tool"
        mock_tool.description = "A tool"
        mock_tool.result_as_answer = False
        mock_tool.args_schema = None

        mock_agent = Mock()
        mock_agent.role = "Researcher"

        mock_crew = Mock()
        mock_crew.before_tool_call = None
        mock_crew.after_tool_call = None
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: my_tool\nAction Input: {}",
            thought="Use the tool",
            tool="my_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="my_tool", arguments={}
                ),
            )
            mp.setattr(
                ToolUsage,
                "use",
                lambda self, calling, text: "Normal result",
            )

            result = execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert result.result == "Normal result"

    def test_crew_both_hooks_together(self):
        """Test that both before and after hooks work together."""
        from crewai.agents.parser import AgentAction
        from crewai.utilities.tool_utils import execute_tool_and_check_finality

        mock_tool = Mock()
        mock_tool.name = "my_tool"
        mock_tool.description = "A tool"
        mock_tool.result_as_answer = False
        mock_tool.args_schema = None

        mock_agent = Mock()
        mock_agent.role = "Admin"

        call_order = []

        def before_tool_call(agent, tool_name, tool_input):
            call_order.append("before")

        def after_tool_call(agent, tool_name, tool_input, tool_output):
            call_order.append("after")

        mock_crew = Mock()
        mock_crew.before_tool_call = before_tool_call
        mock_crew.after_tool_call = after_tool_call
        mock_crew.verbose = False

        action = AgentAction(
            text="Action: my_tool\nAction Input: {}",
            thought="Use the tool",
            tool="my_tool",
            tool_input="{}",
        )

        from crewai.tools.tool_usage import ToolUsage
        from crewai.tools.tool_calling import ToolCalling

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ToolUsage,
                "parse_tool_calling",
                lambda self, text: ToolCalling(
                    tool_name="my_tool", arguments={}
                ),
            )
            mp.setattr(
                ToolUsage,
                "use",
                lambda self, calling, text: "Result",
            )

            result = execute_tool_and_check_finality(
                agent_action=action,
                tools=[mock_tool],
                agent=mock_agent,
                crew=mock_crew,
            )

        assert call_order == ["before", "after"]
        assert result.result == "Result"

    def test_crew_hook_fields_on_crew_model(self):
        """Test that before_tool_call and after_tool_call can be set on Crew."""
        from crewai import Agent, Crew, Task

        def before_hook(agent, tool_name, tool_input):
            pass

        def after_hook(agent, tool_name, tool_input, tool_output):
            pass

        agent = Agent(
            role="Researcher",
            goal="Research",
            backstory="A researcher",
            allow_delegation=False,
        )

        task = Task(
            description="Do research",
            expected_output="Results",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            before_tool_call=before_hook,
            after_tool_call=after_hook,
        )

        assert crew.before_tool_call is not None
        assert crew.after_tool_call is not None

    def test_crew_hooks_default_to_none(self):
        """Test that hooks default to None when not set."""
        from crewai import Agent, Crew, Task

        agent = Agent(
            role="Researcher",
            goal="Research",
            backstory="A researcher",
            allow_delegation=False,
        )

        task = Task(
            description="Do research",
            expected_output="Results",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
        )

        assert crew.before_tool_call is None
        assert crew.after_tool_call is None

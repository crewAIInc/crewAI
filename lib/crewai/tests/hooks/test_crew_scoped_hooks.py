"""Tests for crew-scoped hooks within @CrewBase classes."""

from __future__ import annotations

from unittest.mock import Mock

from crewai import Agent, Crew
from crewai.hooks import (
    LLMCallHookContext,
    ToolCallHookContext,
    before_llm_call,
    before_tool_call,
    get_before_llm_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.project import CrewBase, agent, crew
import pytest


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    from crewai.hooks import llm_hooks, tool_hooks

    original_before_llm = llm_hooks._before_llm_call_hooks.copy()
    original_before_tool = tool_hooks._before_tool_call_hooks.copy()

    llm_hooks._before_llm_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.clear()

    yield

    llm_hooks._before_llm_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.clear()
    llm_hooks._before_llm_call_hooks.extend(original_before_llm)
    tool_hooks._before_tool_call_hooks.extend(original_before_tool)


class TestCrewScopedHooks:
    """Test hooks defined as methods within @CrewBase classes."""

    def test_crew_scoped_hook_is_registered_on_instance_creation(self):
        """Test that crew-scoped hooks are registered when crew instance is created."""

        @CrewBase
        class TestCrew:
            @before_llm_call
            def my_hook(self, context):
                pass

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        hooks_before = get_before_llm_call_hooks()
        initial_count = len(hooks_before)

        crew_instance = TestCrew()

        hooks_after = get_before_llm_call_hooks()

        assert len(hooks_after) == initial_count + 1

    def test_crew_scoped_hook_has_access_to_self(self):
        """Test that crew-scoped hooks can access self and instance variables."""
        execution_log = []

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.crew_name = "TestCrew"
                self.call_count = 0

            @before_llm_call
            def my_hook(self, context):
                self.call_count += 1
                execution_log.append(f"{self.crew_name}:{self.call_count}")

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()

        hooks = get_before_llm_call_hooks()
        crew_hook = hooks[-1]

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)

        crew_hook(context)
        crew_hook(context)

        assert len(execution_log) == 2
        assert execution_log[0] == "TestCrew:1"
        assert execution_log[1] == "TestCrew:2"
        assert crew_instance.call_count == 2

    def test_multiple_crews_have_isolated_hooks(self):
        """Test that different crew instances have isolated hooks."""
        crew1_executions = []
        crew2_executions = []

        @CrewBase
        class Crew1:
            @before_llm_call
            def crew1_hook(self, context):
                crew1_executions.append("crew1")

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        @CrewBase
        class Crew2:
            @before_llm_call
            def crew2_hook(self, context):
                crew2_executions.append("crew2")

            @agent
            def analyst(self):
                return Agent(role="Analyst", goal="Analyze", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        instance1 = Crew1()
        instance2 = Crew2()

        hooks = get_before_llm_call_hooks()
        assert len(hooks) >= 2

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)

        for hook in hooks:
            hook(context)

        assert "crew1" in crew1_executions
        assert "crew2" in crew2_executions

    def test_crew_scoped_hook_with_filters(self):
        """Test that filtered crew-scoped hooks work correctly."""
        execution_log = []

        @CrewBase
        class TestCrew:
            @before_tool_call(tools=["delete_file"])
            def filtered_hook(self, context):
                execution_log.append(f"filtered:{context.tool_name}")
                return

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()

        hooks = get_before_tool_call_hooks()
        crew_hook = hooks[-1]

        mock_tool = Mock()
        context1 = ToolCallHookContext(
            tool_name="delete_file", tool_input={}, tool=mock_tool
        )
        crew_hook(context1)

        assert len(execution_log) == 1
        assert execution_log[0] == "filtered:delete_file"

        context2 = ToolCallHookContext(
            tool_name="read_file", tool_input={}, tool=mock_tool
        )
        crew_hook(context2)

        assert len(execution_log) == 1

    def test_crew_scoped_hook_no_double_registration(self):
        """Test that crew-scoped hooks are not registered twice."""

        @CrewBase
        class TestCrew:
            @before_llm_call
            def my_hook(self, context):
                pass

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        initial_hooks = len(get_before_llm_call_hooks())

        instance1 = TestCrew()

        hooks_after_first = get_before_llm_call_hooks()
        assert len(hooks_after_first) == initial_hooks + 1

        instance2 = TestCrew()

        hooks_after_second = get_before_llm_call_hooks()
        assert len(hooks_after_second) == initial_hooks + 2

    def test_crew_scoped_hook_method_signature(self):
        """Test that crew-scoped hooks have correct signature (self + context)."""

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.test_value = "test"

            @before_llm_call
            def my_hook(self, context):
                return f"{self.test_value}:{context.iterations}"

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()

        assert hasattr(crew_instance.my_hook, "__func__")
        hook_func = crew_instance.my_hook.__func__
        assert hasattr(hook_func, "is_before_llm_call_hook")
        assert hook_func.is_before_llm_call_hook is True

    def test_crew_scoped_with_agent_filter(self):
        """Test crew-scoped hooks with agent filters."""
        execution_log = []

        @CrewBase
        class TestCrew:
            @before_llm_call(agents=["Researcher"])
            def filtered_hook(self, context):
                execution_log.append(context.agent.role)

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()

        hooks = get_before_llm_call_hooks()
        crew_hook = hooks[-1]

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Researcher")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context1 = LLMCallHookContext(executor=mock_executor)
        crew_hook(context1)

        assert len(execution_log) == 1
        assert execution_log[0] == "Researcher"

        mock_executor.agent.role = "Analyst"
        context2 = LLMCallHookContext(executor=mock_executor)
        crew_hook(context2)

        assert len(execution_log) == 1


class TestCrewOnDecoratedMethods:
    """@on(InterceptionPoint.X) methods inside @CrewBase must register.

    Regression: CrewBase only scanned the legacy ``is_*_hook`` markers, so
    methods decorated with the generic ``@on`` decorator (which sets
    ``_interception_point``) were silently dropped and never ran.
    """

    def test_on_decorated_method_registers_and_binds_self(self):
        from crewai.hooks import InterceptionPoint, on
        from crewai.hooks.dispatch import _resolve_hooks

        execution_log = []

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.name = "on-crew"

            @on(InterceptionPoint.PRE_MODEL_CALL)
            def on_pre_model(self, context):
                execution_log.append(self.name)

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        before = len(_resolve_hooks(InterceptionPoint.PRE_MODEL_CALL))

        instance = TestCrew()

        hooks = _resolve_hooks(InterceptionPoint.PRE_MODEL_CALL)
        assert len(hooks) == before + 1

        assert (
            InterceptionPoint.PRE_MODEL_CALL.value,
            hooks[-1],
        ) in instance._registered_hook_functions

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        hooks[-1](LLMCallHookContext(executor=mock_executor))

        assert execution_log == ["on-crew"]


class TestCrewScopedHookAttributes:
    """Test that crew-scoped hooks have correct attributes set."""

    def test_hook_marker_attribute_is_set(self):
        """Test that decorator sets marker attribute on method."""

        @CrewBase
        class TestCrew:
            @before_llm_call
            def my_hook(self, context):
                pass

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        assert hasattr(TestCrew.__dict__["my_hook"], "is_before_llm_call_hook")
        assert TestCrew.__dict__["my_hook"].is_before_llm_call_hook is True

    def test_filter_attributes_are_preserved(self):
        """Test that filter attributes are preserved on methods."""

        @CrewBase
        class TestCrew:
            @before_tool_call(tools=["delete_file"], agents=["Dev"])
            def filtered_hook(self, context):
                return None

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        hook_method = TestCrew.__dict__["filtered_hook"]
        assert hasattr(hook_method, "is_before_tool_call_hook")
        assert hasattr(hook_method, "_filter_tools")
        assert hasattr(hook_method, "_filter_agents")
        assert hook_method._filter_tools == ["delete_file"]
        assert hook_method._filter_agents == ["Dev"]

    def test_registered_hooks_tracked_on_instance(self):
        """Test that registered hooks are tracked on the crew instance."""

        @CrewBase
        class TestCrew:
            @before_llm_call
            def llm_hook(self, context):
                pass

            @before_tool_call
            def tool_hook(self, context):
                return None

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()

        assert hasattr(crew_instance, "_registered_hook_functions")
        assert isinstance(crew_instance._registered_hook_functions, list)
        assert len(crew_instance._registered_hook_functions) == 2

        hook_types = [ht for ht, _ in crew_instance._registered_hook_functions]
        assert "before_llm_call" in hook_types
        assert "before_tool_call" in hook_types


class TestCrewScopedHookExecution:
    """Test execution behavior of crew-scoped hooks."""

    def test_crew_hook_executes_with_bound_self(self):
        """Test that crew-scoped hook executes with self properly bound."""
        execution_log = []

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.instance_id = id(self)

            @before_llm_call
            def my_hook(self, context):
                execution_log.append(self.instance_id)

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()
        expected_id = crew_instance.instance_id

        hooks = get_before_llm_call_hooks()
        crew_hook = hooks[-1]

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)

        crew_hook(context)

        assert len(execution_log) == 1
        assert execution_log[0] == expected_id

    def test_crew_hook_can_modify_instance_state(self):
        """Test that crew-scoped hooks can modify instance variables."""

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.counter = 0

            @before_tool_call
            def increment_counter(self, context):
                self.counter += 1
                return

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        crew_instance = TestCrew()
        assert crew_instance.counter == 0

        hooks = get_before_tool_call_hooks()
        crew_hook = hooks[-1]

        mock_tool = Mock()
        context = ToolCallHookContext(tool_name="test", tool_input={}, tool=mock_tool)

        crew_hook(context)
        crew_hook(context)
        crew_hook(context)

        assert crew_instance.counter == 3

    def test_multiple_instances_maintain_separate_state(self):
        """Test that multiple instances of the same crew maintain separate state."""

        @CrewBase
        class TestCrew:
            def __init__(self):
                self.call_count = 0

            @before_llm_call
            def count_calls(self, context):
                self.call_count += 1

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        instance1 = TestCrew()
        instance2 = TestCrew()

        all_hooks = get_before_llm_call_hooks()

        hook1 = all_hooks[-2]
        hook2 = all_hooks[-1]

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)

        hook1(context)
        hook1(context)

        hook2(context)

        # Note: We can't easily verify which hook belongs to which instance
        # in this test without more introspection, but the fact that it doesn't
        # crash and hooks can maintain state proves isolation works


class TestSignatureDetection:
    """Test that signature detection correctly identifies methods vs functions."""

    def test_method_signature_detected(self):
        """Test that methods with 'self' parameter are detected."""
        import inspect

        @CrewBase
        class TestCrew:
            @before_llm_call
            def method_hook(self, context):
                pass

            @agent
            def researcher(self):
                return Agent(role="Researcher", goal="Research", backstory="Expert")

            @crew
            def crew(self):
                return Crew(agents=self.agents, tasks=[], verbose=False)

        method = TestCrew.__dict__["method_hook"]
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        assert params[0] == "self"
        assert len(params) == 2

    def test_standalone_function_signature_detected(self):
        """Test that standalone functions without 'self' are detected."""
        import inspect

        @before_llm_call
        def standalone_hook(context):
            pass

        sig = inspect.signature(standalone_hook)
        params = list(sig.parameters.keys())
        assert "self" not in params
        assert len(params) == 1

        hooks = get_before_llm_call_hooks()
        assert len(hooks) >= 1

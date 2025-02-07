import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from crewai.traces.context import TraceContext
from crewai.traces.enums import CrewType, RunType, TraceType
from crewai.traces.models import (
    CrewTrace,
    FlowStepIO,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from crewai.traces.unified_trace_controller import (
    UnifiedTraceController,
    init_crew_main_trace,
    init_flow_main_trace,
    should_trace,
    trace_flow_step,
    trace_llm_call,
)


class TestUnifiedTraceController:
    @pytest.fixture
    def basic_trace_controller(self):
        return UnifiedTraceController(
            trace_type=TraceType.LLM_CALL,
            run_type=RunType.KICKOFF,
            crew_type=CrewType.CREW,
            run_id="test-run-id",
            agent_role="test-agent",
            task_name="test-task",
            task_description="test description",
            task_id="test-task-id",
        )

    def test_initialization(self, basic_trace_controller):
        """Test basic initialization of UnifiedTraceController"""
        assert basic_trace_controller.trace_type == TraceType.LLM_CALL
        assert basic_trace_controller.run_type == RunType.KICKOFF
        assert basic_trace_controller.crew_type == CrewType.CREW
        assert basic_trace_controller.run_id == "test-run-id"
        assert basic_trace_controller.agent_role == "test-agent"
        assert basic_trace_controller.task_name == "test-task"
        assert basic_trace_controller.task_description == "test description"
        assert basic_trace_controller.task_id == "test-task-id"
        assert basic_trace_controller.status == "running"
        assert isinstance(UUID(basic_trace_controller.trace_id), UUID)

    def test_start_trace(self, basic_trace_controller):
        """Test starting a trace"""
        result = basic_trace_controller.start_trace()
        assert result == basic_trace_controller
        assert basic_trace_controller.start_time is not None
        assert isinstance(basic_trace_controller.start_time, datetime)

    def test_end_trace_success(self, basic_trace_controller):
        """Test ending a trace successfully"""
        basic_trace_controller.start_trace()
        basic_trace_controller.end_trace(result={"test": "result"})

        assert basic_trace_controller.end_time is not None
        assert basic_trace_controller.status == "completed"
        assert basic_trace_controller.error is None
        assert basic_trace_controller.context.get("response") == {"test": "result"}

    def test_end_trace_with_error(self, basic_trace_controller):
        """Test ending a trace with an error"""
        basic_trace_controller.start_trace()
        basic_trace_controller.end_trace(error="Test error occurred")

        assert basic_trace_controller.end_time is not None
        assert basic_trace_controller.status == "error"
        assert basic_trace_controller.error == "Test error occurred"

    def test_add_child_trace(self, basic_trace_controller):
        """Test adding a child trace"""
        child_trace = {"id": "child-1", "type": "test"}
        basic_trace_controller.add_child_trace(child_trace)
        assert len(basic_trace_controller.children) == 1
        assert basic_trace_controller.children[0] == child_trace

    def test_to_crew_trace_llm_call(self):
        """Test converting to CrewTrace for LLM call"""
        test_messages = [{"role": "user", "content": "test"}]
        test_response = {
            "content": "test response",
            "finish_reason": "stop",
        }

        controller = UnifiedTraceController(
            trace_type=TraceType.LLM_CALL,
            run_type=RunType.KICKOFF,
            crew_type=CrewType.CREW,
            run_id="test-run-id",
            context={
                "messages": test_messages,
                "temperature": 0.7,
                "max_tokens": 100,
            },
        )

        # Set model and messages in the context
        controller.context["model"] = "gpt-4"
        controller.context["messages"] = test_messages

        controller.start_trace()
        controller.end_trace(result=test_response)

        crew_trace = controller.to_crew_trace()
        assert isinstance(crew_trace, CrewTrace)
        assert isinstance(crew_trace.request, LLMRequest)
        assert isinstance(crew_trace.response, LLMResponse)
        assert crew_trace.request.model == "gpt-4"
        assert crew_trace.request.messages == test_messages
        assert crew_trace.response.content == test_response["content"]
        assert crew_trace.response.finish_reason == test_response["finish_reason"]

    def test_to_crew_trace_flow_step(self):
        """Test converting to CrewTrace for flow step"""
        flow_step_data = {
            "function_name": "test_function",
            "inputs": {"param1": "value1"},
            "metadata": {"meta": "data"},
        }

        controller = UnifiedTraceController(
            trace_type=TraceType.FLOW_STEP,
            run_type=RunType.KICKOFF,
            crew_type=CrewType.FLOW,
            run_id="test-run-id",
            flow_step=flow_step_data,
        )

        controller.start_trace()
        controller.end_trace(result="test result")

        crew_trace = controller.to_crew_trace()
        assert isinstance(crew_trace, CrewTrace)
        assert isinstance(crew_trace.flow_step, FlowStepIO)
        assert crew_trace.flow_step.function_name == "test_function"
        assert crew_trace.flow_step.inputs == {"param1": "value1"}
        assert crew_trace.flow_step.outputs == {"result": "test result"}

    def test_should_trace(self):
        """Test should_trace function"""
        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            assert should_trace() is True

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "false"}):
            assert should_trace() is False

        with patch.dict(os.environ, clear=True):
            assert should_trace() is False

    @pytest.mark.asyncio
    async def test_trace_flow_step_decorator(self):
        """Test trace_flow_step decorator"""

        class TestFlow:
            flow_id = "test-flow-id"

            @trace_flow_step
            async def test_method(self, method_name, method, *args, **kwargs):
                return "test result"

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            flow = TestFlow()
            result = await flow.test_method("test_method", lambda x: x, arg1="value1")
            assert result == "test result"

    def test_trace_llm_call_decorator(self):
        """Test trace_llm_call decorator"""

        class TestLLM:
            model = "gpt-4"
            temperature = 0.7
            max_tokens = 100
            stop = None

            def _get_execution_context(self):
                return MagicMock(), MagicMock()

            def _get_new_messages(self, messages):
                return messages

            def _get_new_tool_results(self, agent):
                return []

            @trace_llm_call
            def test_method(self, params):
                return {
                    "choices": [
                        {
                            "message": {"content": "test response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "total_tokens": 50,
                        "prompt_tokens": 20,
                        "completion_tokens": 30,
                    },
                }

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            llm = TestLLM()
            result = llm.test_method({"messages": []})
            assert result["choices"][0]["message"]["content"] == "test response"

    def test_init_crew_main_trace_kickoff(self):
        """Test init_crew_main_trace in kickoff mode"""
        trace_context = None

        class TestCrew:
            id = "test-crew-id"
            _test = False
            _train = False

        @init_crew_main_trace
        def test_method(self):
            nonlocal trace_context
            trace_context = TraceContext.get_current()
            return "test result"

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            crew = TestCrew()
            result = test_method(crew)
            assert result == "test result"
            assert trace_context is not None
            assert trace_context.trace_type == TraceType.LLM_CALL
            assert trace_context.run_type == RunType.KICKOFF
            assert trace_context.crew_type == CrewType.CREW
            assert trace_context.run_id == str(crew.id)

    def test_init_crew_main_trace_test_mode(self):
        """Test init_crew_main_trace in test mode"""
        trace_context = None

        class TestCrew:
            id = "test-crew-id"
            _test = True
            _train = False

        @init_crew_main_trace
        def test_method(self):
            nonlocal trace_context
            trace_context = TraceContext.get_current()
            return "test result"

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            crew = TestCrew()
            result = test_method(crew)
            assert result == "test result"
            assert trace_context is not None
            assert trace_context.run_type == RunType.TEST

    def test_init_crew_main_trace_train_mode(self):
        """Test init_crew_main_trace in train mode"""
        trace_context = None

        class TestCrew:
            id = "test-crew-id"
            _test = False
            _train = True

        @init_crew_main_trace
        def test_method(self):
            nonlocal trace_context
            trace_context = TraceContext.get_current()
            return "test result"

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            crew = TestCrew()
            result = test_method(crew)
            assert result == "test result"
            assert trace_context is not None
            assert trace_context.run_type == RunType.TRAIN

    @pytest.mark.asyncio
    async def test_init_flow_main_trace(self):
        """Test init_flow_main_trace decorator"""
        trace_context = None
        test_inputs = {"test": "input"}

        class TestFlow:
            flow_id = "test-flow-id"

            @init_flow_main_trace
            async def test_method(self, **kwargs):
                nonlocal trace_context
                trace_context = TraceContext.get_current()
                # Verify the context is set during execution
                assert trace_context.context["context"]["inputs"] == test_inputs
                return "test result"

        with patch.dict(os.environ, {"CREWAI_ENABLE_TRACING": "true"}):
            flow = TestFlow()
            result = await flow.test_method(inputs=test_inputs)
            assert result == "test result"
            assert trace_context is not None
            assert trace_context.trace_type == TraceType.FLOW_STEP
            assert trace_context.crew_type == CrewType.FLOW
            assert trace_context.run_type == RunType.KICKOFF
            assert trace_context.run_id == str(flow.flow_id)
            assert trace_context.context["context"]["inputs"] == test_inputs

    def test_trace_context_management(self):
        """Test TraceContext management"""
        trace1 = UnifiedTraceController(
            trace_type=TraceType.LLM_CALL,
            run_type=RunType.KICKOFF,
            crew_type=CrewType.CREW,
            run_id="test-run-1",
        )

        trace2 = UnifiedTraceController(
            trace_type=TraceType.FLOW_STEP,
            run_type=RunType.TEST,
            crew_type=CrewType.FLOW,
            run_id="test-run-2",
        )

        # Test that context is initially empty
        assert TraceContext.get_current() is None

        # Test setting and getting context
        with TraceContext.set_current(trace1):
            assert TraceContext.get_current() == trace1

            # Test nested context
            with TraceContext.set_current(trace2):
                assert TraceContext.get_current() == trace2

            # Test context restoration after nested block
            assert TraceContext.get_current() == trace1

        # Test context cleanup after with block
        assert TraceContext.get_current() is None

    def test_trace_context_error_handling(self):
        """Test TraceContext error handling"""
        trace = UnifiedTraceController(
            trace_type=TraceType.LLM_CALL,
            run_type=RunType.KICKOFF,
            crew_type=CrewType.CREW,
            run_id="test-run",
        )

        # Test that context is properly cleaned up even if an error occurs
        try:
            with TraceContext.set_current(trace):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert TraceContext.get_current() is None

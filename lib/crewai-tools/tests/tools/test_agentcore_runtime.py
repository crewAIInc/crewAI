"""Unit tests for AWS Bedrock AgentCore Runtime adapter."""

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest
from starlette.exceptions import HTTPException

from crewai_tools.aws.bedrock.runtime.base import AgentCoreRuntime


# --- Helpers ---


def make_crew_output(raw="done", json_dict=None, tasks_output=None, token_usage=None):
    """Build a mock CrewOutput."""
    output = MagicMock()
    output.raw = raw
    output.__str__ = lambda self: self.raw
    output.json_dict = json_dict
    output.tasks_output = tasks_output or []
    output.token_usage = token_usage
    return output


# --- _extract_inputs ---


class TestExtractInputs:
    def test_standard_inputs_dict(self):
        result = AgentCoreRuntime._extract_inputs({"inputs": {"topic": "AI"}})
        assert result == {"topic": "AI"}

    def test_prompt_field(self):
        result = AgentCoreRuntime._extract_inputs({"prompt": "hello"})
        assert result == {"input": "hello"}

    def test_message_field(self):
        result = AgentCoreRuntime._extract_inputs({"message": "hello"})
        assert result == {"input": "hello"}

    def test_input_field(self):
        result = AgentCoreRuntime._extract_inputs({"input": "hello"})
        assert result == {"input": "hello"}

    def test_nested_prompt(self):
        result = AgentCoreRuntime._extract_inputs({"input": {"prompt": "hello"}})
        assert result == {"input": "hello"}

    def test_priority_inputs_over_prompt(self):
        payload = {"inputs": {"topic": "AI"}, "prompt": "ignored"}
        result = AgentCoreRuntime._extract_inputs(payload)
        assert result == {"topic": "AI"}

    def test_missing_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({"foo": "bar"})
        assert exc_info.value.status_code == 400

    def test_empty_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({})
        assert exc_info.value.status_code == 400

    def test_empty_string_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({"prompt": ""})
        assert exc_info.value.status_code == 400

    def test_whitespace_only_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({"prompt": "   "})
        assert exc_info.value.status_code == 400

    def test_numeric_value_coerced(self):
        result = AgentCoreRuntime._extract_inputs({"prompt": 42})
        assert result == {"input": "42"}

    def test_strips_whitespace(self):
        result = AgentCoreRuntime._extract_inputs({"prompt": "  hello  "})
        assert result == {"input": "hello"}


# --- Non-streaming handler ---


class TestNonStreamingHandler:
    @pytest.mark.asyncio
    async def test_returns_response_dict(self):
        mock_crew = MagicMock()
        crew_output = make_crew_output(raw="test response")
        mock_crew.copy.return_value = mock_crew
        mock_crew.kickoff.return_value = crew_output

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=False)

        mock_context = MagicMock()
        result = await runtime._non_streaming_handler(
            {"inputs": {"topic": "AI"}}, mock_context
        )

        assert result["response"] == "test response"
        mock_crew.copy.assert_called_once()
        mock_crew.kickoff.assert_called_once_with(inputs={"topic": "AI"})

    @pytest.mark.asyncio
    async def test_with_prompt_input(self):
        mock_crew = MagicMock()
        crew_output = make_crew_output(raw="result")
        mock_crew.copy.return_value = mock_crew
        mock_crew.kickoff.return_value = crew_output

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=False)

        mock_context = MagicMock()
        result = await runtime._non_streaming_handler(
            {"prompt": "tell me about AI"}, mock_context
        )

        assert result["response"] == "result"
        mock_crew.kickoff.assert_called_once_with(inputs={"input": "tell me about AI"})

    @pytest.mark.asyncio
    async def test_includes_json_dict(self):
        mock_crew = MagicMock()
        crew_output = make_crew_output(
            raw="done", json_dict={"key": "value"}
        )
        mock_crew.copy.return_value = mock_crew
        mock_crew.kickoff.return_value = crew_output

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=False)

        mock_context = MagicMock()
        result = await runtime._non_streaming_handler(
            {"inputs": {}}, mock_context
        )

        assert result["json"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_includes_tasks_output(self):
        mock_task = MagicMock()
        mock_task.agent = "Researcher"
        mock_task.description = "Do research"
        mock_task.raw = "research result"

        mock_crew = MagicMock()
        crew_output = make_crew_output(raw="done", tasks_output=[mock_task])
        mock_crew.copy.return_value = mock_crew
        mock_crew.kickoff.return_value = crew_output

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=False)

        mock_context = MagicMock()
        result = await runtime._non_streaming_handler(
            {"inputs": {}}, mock_context
        )

        assert len(result["tasks_output"]) == 1
        assert result["tasks_output"][0]["agent"] == "Researcher"

    @pytest.mark.asyncio
    async def test_includes_token_usage(self):
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_crew = MagicMock()
        crew_output = make_crew_output(raw="done", token_usage=mock_usage)
        mock_crew.copy.return_value = mock_crew
        mock_crew.kickoff.return_value = crew_output

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=False)

        mock_context = MagicMock()
        result = await runtime._non_streaming_handler(
            {"inputs": {}}, mock_context
        )

        assert result["token_usage"]["total_tokens"] == 150


# --- Streaming handler ---


class TestStreamingHandler:
    @pytest.mark.asyncio
    async def test_yields_text_and_done(self):
        from crewai.types.streaming import StreamChunkType

        text_chunk = MagicMock()
        text_chunk.content = "hello world"
        text_chunk.chunk_type = StreamChunkType.TEXT
        text_chunk.agent_role = "Writer"
        text_chunk.task_name = "write"
        text_chunk.tool_call = None

        crew_output = make_crew_output(raw="final answer")

        mock_streaming = MagicMock()
        mock_streaming.__iter__ = MagicMock(return_value=iter([text_chunk]))
        mock_streaming.result = crew_output
        mock_streaming.get_full_text.return_value = "hello world"

        mock_copy = MagicMock()
        mock_copy.kickoff.return_value = mock_streaming

        mock_crew = MagicMock()
        mock_crew.stream = False
        mock_crew.copy.return_value = mock_copy

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=True)

        mock_context = MagicMock()
        collected = []
        async for ev in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(ev)

        # Should have at least a text event and a done event
        text_events = [e for e in collected if e["event"] == "text"]
        done_events = [e for e in collected if e["event"] == "done"]

        assert len(text_events) >= 1
        assert text_events[0]["content"] == "hello world"
        assert text_events[0]["agent_role"] == "Writer"
        assert len(done_events) == 1
        assert done_events[0]["response"] == "final answer"

    @pytest.mark.asyncio
    async def test_yields_tool_call(self):
        from crewai.types.streaming import StreamChunkType

        tool_call = MagicMock()
        tool_call.tool_name = "search"
        tool_call.arguments = '{"query": "AI"}'

        tc_chunk = MagicMock()
        tc_chunk.content = ""
        tc_chunk.chunk_type = StreamChunkType.TOOL_CALL
        tc_chunk.agent_role = "Researcher"
        tc_chunk.task_name = "research"
        tc_chunk.tool_call = tool_call

        crew_output = make_crew_output(raw="done")

        mock_streaming = MagicMock()
        mock_streaming.__iter__ = MagicMock(return_value=iter([tc_chunk]))
        mock_streaming.result = crew_output
        mock_streaming.get_full_text.return_value = ""

        mock_copy = MagicMock()
        mock_copy.kickoff.return_value = mock_streaming

        mock_crew = MagicMock()
        mock_crew.stream = False
        mock_crew.copy.return_value = mock_copy

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=True)

        mock_context = MagicMock()
        collected = []
        async for ev in runtime._streaming_handler(
            {"inputs": {}}, mock_context
        ):
            collected.append(ev)

        tool_events = [e for e in collected if e["event"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_name"] == "search"
        assert tool_events[0]["arguments"] == '{"query": "AI"}'

    @pytest.mark.asyncio
    async def test_does_not_mutate_shared_crew_on_error(self):
        mock_crew = MagicMock()
        mock_crew.stream = False
        mock_copy = MagicMock()
        mock_copy.kickoff.side_effect = RuntimeError("boom")
        mock_crew.copy.return_value = mock_copy

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=True)

        mock_context = MagicMock()
        collected = []
        async for ev in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(ev)

        # Should yield error and never mutate the shared crew
        assert any(e["event"] == "error" for e in collected)
        assert mock_crew.stream is False  # Never touched
        assert mock_copy.stream is True  # Only the copy was modified


# --- Serve / run ---


class TestServe:
    def test_serve_calls_run(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            with patch.object(AgentCoreRuntime, "run") as mock_run:
                AgentCoreRuntime.serve(MagicMock(), port=9090, debug=True)
                mock_run.assert_called_once()


# --- Entrypoint closure ---


class TestEntrypointClosure:
    def test_streaming_entrypoint_is_async_gen_function(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(crew=MagicMock(), stream=True)

            registered_handler = mock_app.entrypoint.call_args[0][0]
            assert inspect.isasyncgenfunction(registered_handler)
            assert not inspect.ismethod(registered_handler)

    def test_non_streaming_entrypoint_is_coroutine_function(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(crew=MagicMock(), stream=False)

            registered_handler = mock_app.entrypoint.call_args[0][0]
            assert inspect.iscoroutinefunction(registered_handler)
            assert not inspect.ismethod(registered_handler)


# --- App property ---


class TestAppProperty:
    def test_exposes_app(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            runtime = AgentCoreRuntime(crew=MagicMock())
            assert runtime.app is mock_app


# --- Constructor passthrough ---


class TestConstructorPassthrough:
    def test_lifespan_passed_to_app(self):
        async def my_lifespan(app):
            yield

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(crew=MagicMock(), lifespan=my_lifespan)

            MockApp.assert_called_once_with(
                debug=False, lifespan=my_lifespan, middleware=None
            )

    def test_middleware_passed_to_app(self):
        from starlette.middleware import Middleware
        from starlette.middleware.gzip import GZipMiddleware

        mw = [Middleware(GZipMiddleware)]

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(crew=MagicMock(), middleware=mw)

            MockApp.assert_called_once_with(
                debug=False, lifespan=None, middleware=mw
            )

    def test_all_app_params_passed(self):
        async def my_lifespan(app):
            yield

        from starlette.middleware import Middleware
        from starlette.middleware.gzip import GZipMiddleware

        mw = [Middleware(GZipMiddleware)]

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            AgentCoreRuntime(
                crew=MagicMock(),
                debug=True,
                lifespan=my_lifespan,
                middleware=mw,
            )

            MockApp.assert_called_once_with(
                debug=True, lifespan=my_lifespan, middleware=mw
            )


# --- crew_output_to_dict ---


class TestCrewOutputToDict:
    def test_basic_output(self):
        output = make_crew_output(raw="hello")
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert result["response"] == "hello"
        assert "json" not in result

    def test_with_json_dict(self):
        output = make_crew_output(raw="done", json_dict={"a": 1})
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert result["json"] == {"a": 1}

    def test_empty_tasks_not_included(self):
        output = make_crew_output(raw="done", tasks_output=[])
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert "tasks_output" not in result

    def test_none_token_usage_not_included(self):
        output = make_crew_output(raw="done", token_usage=None)
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert "token_usage" not in result

    def test_empty_json_dict_not_included(self):
        output = make_crew_output(raw="done", json_dict={})
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert "json" not in result

    def test_task_missing_attributes_uses_defaults(self):
        mock_task = MagicMock(spec=[])  # spec=[] means no attributes
        output = make_crew_output(raw="done", tasks_output=[mock_task])
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert result["tasks_output"][0]["agent"] == ""
        assert result["tasks_output"][0]["description"] == ""
        assert result["tasks_output"][0]["raw"] == ""

    def test_token_usage_missing_fields_uses_defaults(self):
        usage = MagicMock(spec=[])  # no attributes
        output = make_crew_output(raw="done", token_usage=usage)
        result = AgentCoreRuntime._crew_output_to_dict(output)
        assert result["token_usage"]["prompt_tokens"] == 0
        assert result["token_usage"]["completion_tokens"] == 0
        assert result["token_usage"]["total_tokens"] == 0


# --- _stream_chunk_to_dict ---


class TestStreamChunkToDict:
    def test_text_chunk(self):
        from crewai.types.streaming import StreamChunkType

        chunk = MagicMock()
        chunk.content = "hello"
        chunk.chunk_type = StreamChunkType.TEXT
        chunk.agent_role = "Writer"
        chunk.task_name = "write"
        chunk.tool_call = None

        result = AgentCoreRuntime._stream_chunk_to_dict(chunk)
        assert result == {
            "event": "text",
            "content": "hello",
            "agent_role": "Writer",
            "task_name": "write",
        }

    def test_tool_call_chunk(self):
        from crewai.types.streaming import StreamChunkType

        tc = MagicMock()
        tc.tool_name = "search"
        tc.arguments = '{"q": "AI"}'

        chunk = MagicMock()
        chunk.chunk_type = StreamChunkType.TOOL_CALL
        chunk.tool_call = tc
        chunk.agent_role = "Researcher"
        chunk.task_name = "research"

        result = AgentCoreRuntime._stream_chunk_to_dict(chunk)
        assert result["event"] == "tool_call"
        assert result["tool_name"] == "search"
        assert result["arguments"] == '{"q": "AI"}'

    def test_tool_call_with_none_tool_call_falls_to_text(self):
        from crewai.types.streaming import StreamChunkType

        chunk = MagicMock()
        chunk.chunk_type = StreamChunkType.TOOL_CALL
        chunk.tool_call = None  # edge case
        chunk.content = "fallback"
        chunk.agent_role = "Agent"
        chunk.task_name = "task"

        result = AgentCoreRuntime._stream_chunk_to_dict(chunk)
        assert result["event"] == "text"
        assert result["content"] == "fallback"

    def test_tool_call_with_none_tool_name(self):
        from crewai.types.streaming import StreamChunkType

        tc = MagicMock()
        tc.tool_name = None
        tc.arguments = "{}"

        chunk = MagicMock()
        chunk.chunk_type = StreamChunkType.TOOL_CALL
        chunk.tool_call = tc
        chunk.agent_role = "Agent"
        chunk.task_name = "task"

        result = AgentCoreRuntime._stream_chunk_to_dict(chunk)
        assert result["event"] == "tool_call"
        assert result["tool_name"] == ""


# --- _extract_inputs edge cases ---


class TestExtractInputsEdgeCases:
    def test_nested_dict_without_prompt_key_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({"input": {"other": "value"}})
        assert exc_info.value.status_code == 400

    def test_float_value_coerced(self):
        result = AgentCoreRuntime._extract_inputs({"prompt": 3.14})
        assert result == {"input": "3.14"}

    def test_empty_prompt_falls_to_message(self):
        result = AgentCoreRuntime._extract_inputs({"prompt": "", "message": "hello"})
        assert result == {"input": "hello"}

    def test_list_value_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            AgentCoreRuntime._extract_inputs({"prompt": ["a", "b"]})
        assert exc_info.value.status_code == 400

    def test_inputs_non_dict_falls_to_prompt(self):
        result = AgentCoreRuntime._extract_inputs({"inputs": "not a dict", "prompt": "hi"})
        assert result == {"input": "hi"}


# --- run() passthrough ---


class TestRunPassthrough:
    def test_run_passes_port_and_host(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            runtime = AgentCoreRuntime(
                crew=MagicMock(), port=9090, host="0.0.0.0"
            )
            runtime.run()

            mock_app.run.assert_called_once_with(port=9090, host="0.0.0.0")

    def test_run_passes_extra_kwargs(self):
        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ) as MockApp:
            mock_app = MagicMock()
            MockApp.return_value = mock_app

            runtime = AgentCoreRuntime(crew=MagicMock())
            runtime.run(workers=4)

            mock_app.run.assert_called_once_with(port=8080, host=None, workers=4)


# --- Streaming RuntimeError fallback ---


class TestStreamingFallback:
    @pytest.mark.asyncio
    async def test_result_runtime_error_falls_back_to_full_text(self):
        from crewai.types.streaming import StreamChunkType

        text_chunk = MagicMock()
        text_chunk.content = "partial"
        text_chunk.chunk_type = StreamChunkType.TEXT
        text_chunk.agent_role = "Writer"
        text_chunk.task_name = "write"
        text_chunk.tool_call = None

        mock_streaming = MagicMock()
        mock_streaming.__iter__ = MagicMock(return_value=iter([text_chunk]))
        mock_streaming.get_full_text.return_value = "full text here"

        # .result raises RuntimeError (e.g., result not yet available)
        type(mock_streaming).result = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("not ready"))
        )

        mock_copy = MagicMock()
        mock_copy.kickoff.return_value = mock_streaming

        mock_crew = MagicMock()
        mock_crew.copy.return_value = mock_copy

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=True)

        mock_context = MagicMock()
        collected = []
        async for ev in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(ev)

        done_events = [e for e in collected if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["response"] == "full text here"

    @pytest.mark.asyncio
    async def test_error_during_chunk_iteration(self):
        mock_streaming = MagicMock()

        def exploding_iter(self):
            raise ValueError("chunk decode error")

        mock_streaming.__iter__ = exploding_iter

        mock_copy = MagicMock()
        mock_copy.kickoff.return_value = mock_streaming

        mock_crew = MagicMock()
        mock_crew.copy.return_value = mock_copy

        with patch(
            "crewai_tools.aws.bedrock.runtime.base.BedrockAgentCoreApp"
        ):
            runtime = AgentCoreRuntime(crew=mock_crew, stream=True)

        mock_context = MagicMock()
        collected = []
        async for ev in runtime._streaming_handler(
            {"prompt": "hello"}, mock_context
        ):
            collected.append(ev)

        error_events = [e for e in collected if e["event"] == "error"]
        assert len(error_events) == 1
        assert "chunk decode error" in error_events[0]["message"]


# --- Top-level export ---


class TestExport:
    def test_importable_from_bedrock_package(self):
        from crewai_tools.aws.bedrock import AgentCoreRuntime as Exported

        assert Exported is AgentCoreRuntime

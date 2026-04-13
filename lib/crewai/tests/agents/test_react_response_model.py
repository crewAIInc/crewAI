"""Tests for output_pydantic behavior in ReAct mode (no function calling).

Regression test for https://github.com/crewAIInc/crewAI/issues/4695
When a Task has output_pydantic set and the LLM does NOT support native
function calling, the executor falls back to the ReAct text-based loop.
Previously, response_model was still forwarded to the LLM call, which caused
instructor to inject the Pydantic schema as a tool — something models
without function-calling support cannot handle.

The fix: _invoke_loop_react() must pass response_model=None so the LLM
produces plain text that the ReAct parser can handle normally.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor


@pytest.fixture
def mock_llm():
    """Create an LLM mock that does NOT support function calling."""
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.supports_function_calling.return_value = False
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task():
    task = MagicMock()
    task.description = "Test task"
    return task


@pytest.fixture
def mock_crew():
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    return crew


def _make_executor(mock_llm, mock_agent, mock_task, mock_crew, response_model):
    """Build a CrewAgentExecutor with the given response_model."""
    return CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
        prompt={"prompt": "Test {input} {tool_names} {tools}"},
        max_iter=3,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=MagicMock(),
        response_model=response_model,
    )


class TestReActResponseModel:
    """Ensure response_model is NOT forwarded in ReAct mode."""

    def test_react_path_passes_response_model_none(
        self, mock_llm, mock_agent, mock_task, mock_crew
    ):
        """When LLM lacks function-calling, _invoke_loop_react must NOT send
        response_model to get_llm_response.
        """
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        executor = _make_executor(
            mock_llm, mock_agent, mock_task, mock_crew,
            response_model=MyOutput,
        )
        # The executor should store the response_model
        assert executor.response_model is MyOutput

        final_text = 'Final Answer: {"answer": "hello"}'

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value=final_text,
        ) as mock_get:
            result = executor._invoke_loop_react()

            # The critical assertion: response_model passed to get_llm_response
            # MUST be None — never the Pydantic model — in ReAct mode.
            call_kwargs = mock_get.call_args.kwargs
            assert call_kwargs["response_model"] is None, (
                "response_model must be None in ReAct mode to prevent "
                "instructor from injecting the schema as a tool"
            )

    def test_react_path_selected_when_no_function_calling(
        self, mock_llm, mock_agent, mock_task, mock_crew
    ):
        """Verify the executor chooses _invoke_loop_react when
        supports_function_calling() returns False."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            value: int

        executor = _make_executor(
            mock_llm, mock_agent, mock_task, mock_crew,
            response_model=MyOutput,
        )

        final_text = 'Final Answer: {"value": 42}'

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value=final_text,
        ):
            with patch.object(
                executor, "_invoke_loop_react", wraps=executor._invoke_loop_react
            ) as spy_react, patch.object(
                executor, "_invoke_loop_native_tools"
            ) as spy_native:
                executor.invoke({"input": "test", "tool_names": "", "tools": ""})

                spy_react.assert_called_once()
                spy_native.assert_not_called()

    def test_react_path_with_valid_json_string_bypasses_fallback_parser(
        self, mock_llm, mock_agent, mock_task, mock_crew
    ):
        """Valid JSON strings should be returned directly in ReAct mode."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        executor = _make_executor(
            mock_llm, mock_agent, mock_task, mock_crew, response_model=MyOutput
        )

        valid_json = '{"answer": "hello"}'

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value=valid_json,
        ), patch(
            "crewai.agents.crew_agent_executor.process_llm_response"
        ) as mock_process:
            result = executor._invoke_loop_react()

        mock_process.assert_not_called()
        assert result.output == valid_json
        assert result.text == valid_json

    @pytest.mark.asyncio
    async def test_async_react_path_with_valid_json_string_bypasses_fallback_parser(
        self, mock_llm, mock_agent, mock_task, mock_crew
    ):
        """Async ReAct mode should treat valid JSON strings the same way."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        executor = _make_executor(
            mock_llm, mock_agent, mock_task, mock_crew, response_model=MyOutput
        )

        valid_json = '{"answer": "hello"}'

        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new=AsyncMock(return_value=valid_json),
        ), patch(
            "crewai.agents.crew_agent_executor.process_llm_response"
        ) as mock_process:
            result = await executor._ainvoke_loop_react()

        mock_process.assert_not_called()
        assert result.output == valid_json
        assert result.text == valid_json

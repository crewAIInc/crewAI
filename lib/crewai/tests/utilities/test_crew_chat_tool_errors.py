"""Regression tests: a failing crew run inside `crewai chat` must surface the
error to the conversation instead of killing the whole session.

``run_crew_tool`` is the LLM-callable tool behind the chat loop (declared
``-> str``, invoked via ``chat_llm.call(available_functions=...)``). It used
to call ``sys.exit(1)`` on failure — ``SystemExit`` is a ``BaseException``, so
``chat_loop``'s ``except Exception`` could not catch it and the entire
interactive session died mid-conversation, with the model never seeing the
error.
"""

from typing import Any
from unittest.mock import Mock, patch

import crewai.utilities.crew_chat as crew_chat


def test_crew_tool_failure_returns_error_to_conversation() -> None:
    """A failing kickoff returns an error string the LLM can relay (fails on
    main: sys.exit(1) raises SystemExit and kills the session)."""
    crew = Mock()
    crew.kickoff.side_effect = RuntimeError("provider 503")

    result = crew_chat.run_crew_tool(crew, [{"role": "user", "content": "hi"}], topic="AI")

    assert isinstance(result, str)
    assert "provider 503" in result


def test_crew_tool_success_returns_output() -> None:
    """Guard: a successful run still returns the crew output as a string."""
    crew = Mock()
    crew.kickoff.return_value = "final crew output"

    result = crew_chat.run_crew_tool(crew, [{"role": "user", "content": "hi"}], topic="AI")

    assert result == "final crew output"
    crew.kickoff.assert_called_once()
    passed_inputs = crew.kickoff.call_args.kwargs["inputs"]
    assert "crew_chat_messages" in passed_inputs


def test_chat_survives_a_failing_tool_call() -> None:
    """End to end over handle_user_input: the model invokes the tool, the run
    fails, the error lands in the conversation and the chat keeps going."""
    crew = Mock()
    crew.kickoff.side_effect = RuntimeError("provider 503")
    tool_name = "my_crew"
    available_functions = {
        tool_name: crew_chat.create_tool_function(crew, [])
    }

    class ToolCallingLLM:
        def call(self, messages: Any, tools: Any = None, available_functions: Any = None) -> str:
            tool_result = available_functions[tool_name](topic="AI")
            return f"Assistant: I couldn't run the crew ({tool_result})"

    chat_llm = ToolCallingLLM()
    messages: list[Any] = []

    with patch.object(crew_chat.click, "echo"):
        crew_chat.handle_user_input(
            "please run the crew",
            chat_llm=chat_llm,
            messages=messages,
            crew_tool_schema={"type": "function", "function": {"name": tool_name}},
            available_functions=available_functions,
        )

    # No SystemExit escaped, the reply landed in the conversation, and the
    # user's next turn would reach the model with the error context intact.
    assert len(messages) == 2  # user message + assistant reply
    assert "provider 503" in messages[-1]["content"]

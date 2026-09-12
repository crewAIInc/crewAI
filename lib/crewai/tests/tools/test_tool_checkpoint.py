from typing import Any
import warnings

from crewai import Agent, Crew, Task
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.checkpoint_listener import _do_checkpoint
from crewai.state.runtime import RuntimeState
from crewai.tools import tool as tool_decorator


def sample_top_level_tool_func(text: str) -> str:
    """A top level named tool function."""
    return text


def test_tool_checkpoint_json_serialization(tmp_path):
    @tool_decorator("echo_tool")
    def echo_tool(text: str) -> str:
        return text

    agent = Agent(role="researcher", goal="research", backstory="backstory", tools=[echo_tool])
    task = Task(description="task desc", expected_output="output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    state = RuntimeState([crew])

    cfg = CheckpointConfig(location=str(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _do_checkpoint(state, cfg, event=None)

    # Should checkpoint cleanly without raising an exception
    assert any("Tool func" in str(w.message) or "guardrail" in str(w.message).lower() for w in caught)

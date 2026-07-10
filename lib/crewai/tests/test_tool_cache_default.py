# mypy: ignore-errors
"""Regression tests for EPD-180: tool-result caching used to be ON by default,
so an LLM calling the same tool with identical arguments twice in one run got
the first (possibly stale) result back without the tool executing — silently
wrong for live-data tools, and silently dropped actions for stateful tools.

Caching is now opt-in: ``Crew(cache=True)`` for crews, ``Agent(cache=True)``
(or an explicit ``cache_handler``) for standalone agents. The machinery —
including per-tool ``cache_function`` write gating — is unchanged once opted
in.

The end-to-end tests run fully offline: a fake OpenAI client scripts two
identical tool calls followed by a final answer, mirroring the EPD-180
clean-room repro.
"""

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from crewai import LLM, Agent, Crew, Task
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools import BaseTool


class LookupArgs(BaseModel):
    city: str = Field(description="City to look up.")


def make_live_tool():
    """A tool returning a different value on every real execution."""
    executions = []

    class LiveLookupTool(BaseTool):
        name: str = "live_lookup"
        description: str = "Returns a live (time-varying) reading for a city."
        args_schema: type[BaseModel] = LookupArgs
        # cache_function deliberately NOT set — exercising the default.

        def _run(self, city: str) -> str:
            executions.append(city)
            return f"reading #{len(executions)} for {city}"

    return LiveLookupTool(), executions


def make_scripted_llm():
    """An offline LLM whose client scripts two identical tool calls."""

    def tool_call_response(call_id: str):
        return {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "live_lookup",
                            "arguments": '{"city": "paris"}',
                        },
                    }
                ],
            },
        }

    scripted = [
        tool_call_response("call_1"),
        tool_call_response("call_2"),  # identical name+args, new id
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Final answer: done."},
        },
    ]

    class FakeCompletions:
        def __init__(self):
            self.n = 0

        def create(self, **params):
            choice = scripted[min(self.n, len(scripted) - 1)]
            self.n += 1
            return ChatCompletion.model_validate(
                {
                    "id": f"chatcmpl-fake-{self.n}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": params.get("model", "gpt-4o"),
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            )

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    llm = LLM(model="openai/gpt-4o")
    llm._client = FakeClient()
    return llm


def run_crew(**crew_kwargs):
    tool, executions = make_live_tool()
    agent = Agent(
        role="reader",
        goal="Look things up.",
        backstory="Test agent.",
        llm=make_scripted_llm(),
        tools=[tool],
        verbose=False,
    )
    task = Task(
        description="Look up paris twice and report.",
        expected_output="A report.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False, **crew_kwargs)
    crew.kickoff()
    return executions


class TestToolCachingIsOptIn:
    def test_default_reexecutes_identical_tool_calls(self):
        """EPD-180: with no opt-in, both identical calls must really execute."""
        executions = run_crew()
        assert len(executions) == 2

    def test_crew_cache_true_dedupes_identical_tool_calls(self):
        """Opting in via Crew(cache=True) restores the dedup behavior."""
        executions = run_crew(cache=True)
        assert len(executions) == 1


class TestAgentCacheWiring:
    def _agent(self, **kwargs) -> Agent:
        return Agent(
            role="reader",
            goal="Look things up.",
            backstory="Test agent.",
            **kwargs,
        )

    def test_standalone_agent_has_no_cache_by_default(self):
        agent = self._agent()
        assert agent.tools_handler.cache is None
        assert agent.cache_handler is None

    def test_standalone_agent_explicit_cache_true_opts_in(self):
        agent = self._agent(cache=True)
        assert agent.tools_handler.cache is not None
        assert agent.cache_handler is not None

    def test_standalone_agent_explicit_cache_handler_opts_in(self):
        handler = CacheHandler()
        agent = self._agent(cache_handler=handler)
        assert agent.tools_handler.cache is handler

    def test_explicit_cache_false_stays_off_even_with_handler(self):
        agent = self._agent(cache=False, cache_handler=CacheHandler())
        assert agent.tools_handler.cache is None

    def test_agents_accept_a_crew_offered_handler_by_default(self):
        """``Crew(cache=True)`` offers its handler via set_cache_handler at
        kickoff; agents that didn't explicitly opt out must accept it."""
        agent = self._agent()
        assert agent.tools_handler.cache is None

        handler = CacheHandler()
        agent.set_cache_handler(handler)
        assert agent.tools_handler.cache is handler

    def test_agents_that_opted_out_refuse_a_crew_offered_handler(self):
        agent = self._agent(cache=False)
        agent.set_cache_handler(CacheHandler())
        assert agent.tools_handler.cache is None

    def test_copy_of_default_agent_does_not_opt_in(self):
        """copy() rebuilds from model_dump(), which includes the field
        default cache=True — that must not read as an explicit opt-in on
        the copy (Bugbot review finding on the original PR)."""
        copied = self._agent().copy()
        assert copied.tools_handler.cache is None
        assert copied.cache_handler is None

    def test_copy_of_opted_in_agent_stays_opted_in(self):
        copied = self._agent(cache=True).copy()
        assert copied.tools_handler.cache is not None

    def test_copy_of_handler_opted_in_agent_stays_opted_in(self):
        """An explicit cache_handler is an opt-in too; copy() excludes the
        handler itself, but the consent must survive — the copy wires its
        own fresh handler (Bugbot review finding on the original PR)."""
        source = self._agent(cache_handler=CacheHandler())
        copied = source.copy()
        assert copied.tools_handler.cache is not None
        assert copied.tools_handler.cache is not source.tools_handler.cache

    def test_copy_of_explicit_cache_false_with_handler_stays_off(self):
        copied = self._agent(cache=False, cache_handler=CacheHandler()).copy()
        assert copied.tools_handler.cache is None


class TestHierarchicalManagerCacheWiring:
    """The auto-created hierarchical manager is built outside the agents
    loop that offers the crew's cache handler; an opted-in crew must wire
    the manager too (Bugbot review finding on the original PR)."""

    def _crew(self, **crew_kwargs) -> Crew:
        from crewai.process import Process

        agent = Agent(role="worker", goal="Do work.", backstory="Test agent.")
        task = Task(description="Do the work.", expected_output="Done.")
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.hierarchical,
            manager_llm="gpt-4o",
            **crew_kwargs,
        )

    def test_manager_gets_crew_handler_when_cache_enabled(self):
        crew = self._crew(cache=True)
        crew._create_manager_agent()
        assert crew.manager_agent.tools_handler.cache is crew._cache_handler

    def test_manager_has_no_cache_when_crew_did_not_opt_in(self):
        crew = self._crew()
        crew._create_manager_agent()
        assert crew.manager_agent.tools_handler.cache is None

    def test_user_provided_manager_with_cache_false_stays_excluded(self):
        manager = Agent(
            role="manager",
            goal="Manage.",
            backstory="Test manager.",
            cache=False,
            allow_delegation=True,
        )
        crew = self._crew(cache=True)
        crew.manager_agent = manager
        crew._create_manager_agent()
        assert manager.tools_handler.cache is None

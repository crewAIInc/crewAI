"""`llm_overlay` swaps an agent's model by role, for the calling context only.

The overlay is read in exactly three places: the validators where `Agent` and
`LiteAgent` resolve their `llm`, and `Agent.interpolate_inputs`, which looks the
interpolated role up again because a templated role only becomes a key once a
kickoff fills its placeholders in. So these tests build agents, interpolate
them, and look at the model they end up with. No LLM is ever called.

`create_llm("openai/gpt-4o")` returns the native OpenAI provider, which strips
the `openai/` prefix, so the resolved model reads `"gpt-4o"`.
"""

from __future__ import annotations

import contextvars
import threading

from crewai import Agent, Crew, Task
from crewai.lite_agent import LiteAgent
from crewai.llm_overlay import active, llm_overlay, overlay_model_for
import pytest


OVERLAY = {"Researcher": "openai/gpt-4o"}
TEMPLATE_OVERLAY = {"Researcher for crewAIInc/x": "openai/gpt-4o"}


def _agent(role: str) -> Agent:
    return Agent(role=role, goal="g", backstory="b", llm="openai/gpt-4o-mini")


def test_matching_role_gets_the_overlay_model_others_keep_their_own() -> None:
    with llm_overlay(OVERLAY):
        researcher = _agent("Researcher")
        writer = _agent("Writer")

    assert researcher.llm.model == "gpt-4o"
    assert writer.llm.model == "gpt-4o-mini"


def test_overlay_does_not_leak_past_the_block() -> None:
    with llm_overlay(OVERLAY):
        assert overlay_model_for("Researcher") == "openai/gpt-4o"

    assert active.get() is None
    assert overlay_model_for("Researcher") is None
    assert _agent("Researcher").llm.model == "gpt-4o-mini"


def test_overlay_is_reset_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), llm_overlay(OVERLAY):
        raise RuntimeError("boom")

    assert active.get() is None


def test_nested_overlay_restores_the_outer_one() -> None:
    with llm_overlay(OVERLAY):
        with llm_overlay(None):
            assert overlay_model_for("Researcher") is None
        assert overlay_model_for("Researcher") == "openai/gpt-4o"


@pytest.mark.filterwarnings("ignore:LiteAgent is deprecated")
def test_lite_agent_gets_the_overlay_model() -> None:
    with llm_overlay(OVERLAY):
        agent = LiteAgent(
            role="Researcher", goal="g", backstory="b", llm="openai/gpt-4o-mini"
        )

    assert agent.llm.model == "gpt-4o"


def test_overlay_does_not_cross_plain_threads_unless_context_is_copied() -> None:
    """Pins contextvar semantics; callers threading agents must copy the context."""
    seen: dict[str, dict[str, str] | None] = {}

    def record(key: str) -> None:
        seen[key] = active.get()

    with llm_overlay(OVERLAY):
        plain = threading.Thread(target=record, args=("plain",))
        plain.start()
        plain.join()

        ctx = contextvars.copy_context()
        copied = threading.Thread(target=ctx.run, args=(record, "copied"))
        copied.start()
        copied.join()

    assert seen["plain"] is None
    assert seen["copied"] == OVERLAY


def test_templated_role_matches_once_its_inputs_are_interpolated() -> None:
    """The template is not a key at construction; the interpolated role is."""
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        assert agent.llm.model == "gpt-4o-mini"

        agent.interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm.model == "gpt-4o"


def test_interpolating_to_a_role_that_is_not_a_key_leaves_the_llm_alone() -> None:
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        agent.interpolate_inputs({"repo": "crewAIInc/other"})

    assert agent.role == "Researcher for crewAIInc/other"
    assert agent.llm.model == "gpt-4o-mini"


def test_interpolating_outside_any_block_leaves_the_llm_alone() -> None:
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")

    agent.interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm.model == "gpt-4o-mini"


def test_a_template_role_that_is_itself_a_key_still_matches_at_construction() -> None:
    with llm_overlay({"Researcher for {repo}": "openai/gpt-4o"}):
        agent = _agent("Researcher for {repo}")

    assert agent.llm.model == "gpt-4o"


def test_each_interpolation_resolves_from_the_template_and_never_reverts() -> None:
    """Each kickoff re-interpolates the original template; the model follows the role.

    While the new role is a key it gets the mapped model; when it is not, the
    agent keeps the model it already has. The overlay only ever sets a model,
    it never restores the declared one.
    """
    mapping = {
        "Researcher for crewAIInc/x": "openai/gpt-4o",
        "Researcher for crewAIInc/y": "openai/gpt-4.1",
    }
    with llm_overlay(mapping):
        agent = _agent("Researcher for {repo}")

        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o"

        agent.interpolate_inputs({"repo": "crewAIInc/y"})
        assert agent.role == "Researcher for crewAIInc/y"
        assert agent.llm.model == "gpt-4.1"

        agent.interpolate_inputs({"repo": "crewAIInc/z"})
        assert agent.role == "Researcher for crewAIInc/z"
        assert agent.llm.model == "gpt-4.1"


def test_interpolating_with_no_inputs_does_not_touch_the_llm() -> None:
    """`interpolate_inputs({})` rewrites nothing, so the overlay is not consulted."""
    with llm_overlay({"Researcher for {repo}": "openai/gpt-4o"}):
        agent = _agent("Writer for {repo}")
        agent.interpolate_inputs({})

    assert agent.role == "Writer for {repo}"
    assert agent.llm.model == "gpt-4o-mini"


def test_a_role_the_interpolation_leaves_unchanged_is_not_resolved_again() -> None:
    """Construction's resolution stands: same model, same llm instance.

    Without this an agent built OUTSIDE the block would pick the mapped model
    up at a kickoff with inputs, and one built inside would lose the state set
    on its llm between construction and kickoff.
    """
    with llm_overlay(OVERLAY):
        inside = _agent("Researcher")
    outside = _agent("Researcher")
    with llm_overlay(OVERLAY):
        inside_llm, outside_llm = inside.llm, outside.llm
        inside.interpolate_inputs({"topic": "ai"})
        outside.interpolate_inputs({"topic": "ai"})

    assert inside.llm is inside_llm and inside.llm.model == "gpt-4o"
    assert outside.llm is outside_llm and outside.llm.model == "gpt-4o-mini"


def test_the_streaming_flag_survives_the_kickoff_time_swap() -> None:
    """`Crew.kickoff(stream=True)` sets `agent.llm.stream` before it interpolates."""
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        agent.llm.stream = True
        agent.interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.llm.model == "gpt-4o" and agent.llm.stream is True


def test_prepare_kickoff_binds_the_executor_to_the_re_resolved_llm() -> None:
    """The real kickoff ordering: interpolate, then set up the agents' executors."""
    from crewai.crews.utils import prepare_kickoff

    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        task = Task(description="Map {repo}", expected_output="a map", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        prepare_kickoff(crew, {"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm.model == "gpt-4o"
    assert agent.agent_executor is not None and agent.agent_executor.llm is agent.llm
    # Every task re-binds the executor to agent.llm (`_update_executor_parameters`).
    agent.create_agent_executor()
    assert agent.agent_executor.llm is agent.llm


def test_crew_input_interpolation_routes_the_templated_role() -> None:
    """The kickoff path: Crew._interpolate_inputs is what rewrites agent roles."""
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        task = Task(description="Map {repo}", expected_output="a map", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])

        crew._interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm.model == "gpt-4o"


def test_re_validation_of_an_existing_agent_does_not_read_the_overlay_again() -> None:
    """The event bus registers an agent in its RuntimeState the first time it
    emits, which re-runs `post_init_setup` on the same object. Inside a block
    that maps the agent's role that used to replace an llm the agent already
    ran on — an agent built OUTSIDE the block picked the mapped model up on its
    first standalone kickoff inside one, and lost `stream=True` with it."""
    from crewai import RuntimeState

    outside = _agent("Researcher")
    outside.llm.stream = True
    outside_llm = outside.llm
    with llm_overlay(OVERLAY):
        inside = _agent("Researcher")
        inside_llm = inside.llm
        state = RuntimeState(root=[outside, inside])

    assert state.root[0] is outside and state.root[1] is inside
    assert (
        outside.llm is outside_llm
        and outside.llm.model == "gpt-4o-mini"
        and outside.llm.stream is True
    )
    assert inside.llm is inside_llm and inside.llm.model == "gpt-4o"


def test_re_validation_keeps_the_llm_a_kickoff_time_swap_set() -> None:
    from crewai import RuntimeState

    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        swapped = agent.llm
        assert swapped.model == "gpt-4o"
        RuntimeState(root=[agent])

    assert agent.llm is swapped

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
from typing import Any

from crewai import Agent, Crew, Task
from crewai.lite_agent import LiteAgent
from crewai.llm import LLM
from crewai.llm_overlay import active, llm_overlay, overlay_model_for
from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion
import pytest


OVERLAY = {"Researcher": "openai/gpt-4o"}
TEMPLATE_OVERLAY = {"Researcher for crewAIInc/x": "openai/gpt-4o"}

# What a declared LLM carries beyond its model: an endpoint, a key, and
# generation settings. `create_llm("<mapped model>")` would have none of them.
CONFIGURATION: dict[str, Any] = {
    "base_url": "http://localhost:9999/v1",
    "api_key": "k",
    "timeout": 42,
    "temperature": 0.1,
    "max_tokens": 77,
}


def _agent(role: str) -> Agent:
    return Agent(role=role, goal="g", backstory="b", llm="openai/gpt-4o-mini")


def _configured_llm() -> LLM:
    return LLM(model="openai/gpt-4o-mini", **CONFIGURATION)


def _configuration_of(llm: Any) -> dict[str, Any]:
    return {name: getattr(llm, name) for name in CONFIGURATION}


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
        declared = agent.llm
        agent.interpolate_inputs({"repo": "crewAIInc/other"})

    assert agent.role == "Researcher for crewAIInc/other"
    assert agent.llm is declared and agent.llm.model == "gpt-4o-mini"


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


def test_each_interpolation_resolves_from_the_template() -> None:
    """Each kickoff re-interpolates the original template; the model follows the role.

    While the new role is a key it gets that key's model; when it is not, the
    agent is back on the llm it was declared with — not on the previous
    role's model.
    """
    mapping = {
        "Researcher for crewAIInc/x": "openai/gpt-4o",
        "Researcher for crewAIInc/y": "openai/gpt-4.1",
    }
    with llm_overlay(mapping):
        agent = _agent("Researcher for {repo}")
        declared = agent.llm

        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o"

        agent.interpolate_inputs({"repo": "crewAIInc/y"})
        assert agent.role == "Researcher for crewAIInc/y"
        assert agent.llm.model == "gpt-4.1"

        agent.interpolate_inputs({"repo": "crewAIInc/z"})
        assert agent.role == "Researcher for crewAIInc/z"
        assert agent.llm is declared and agent.llm.model == "gpt-4o-mini"


def test_a_reused_crew_kicked_off_for_another_input_reverts_to_the_declared_llm() -> (
    None
):
    """One `Crew` object, several kickoffs with different inputs, one block.

    The mapped model applies while the interpolated role is a key. An input
    that interpolates to a role the overlay says nothing about puts the very
    declared instance back — the crew must not keep billing the previous
    input's provider — and the next input that is a key maps again.
    """
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        declared = agent.llm

        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o"

        agent.interpolate_inputs({"repo": "crewAIInc/y"})
        assert agent.role == "Researcher for crewAIInc/y"
        assert agent.llm is declared and agent.llm.model == "gpt-4o-mini"

        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o"


def test_a_template_key_stops_matching_once_the_role_is_interpolated() -> None:
    """The overlay maps the role's current text: the template is a key, the
    interpolated text is not, so a kickoff inside the block puts the declared
    llm back. Keys are written for the interpolated role — the one traces record."""
    with llm_overlay({"Researcher for {repo}": "openai/gpt-4o"}):
        agent = _agent("Researcher for {repo}")
        assert agent.llm.model == "gpt-4o"

        agent.interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm.model == "gpt-4o-mini"


def test_outside_any_block_an_interpolation_that_changes_the_role_changes_nothing() -> (
    None
):
    """Block semantics: with no overlay active there is nothing to re-resolve
    against, so an agent built inside a block keeps the model it got there."""
    with llm_overlay({"Researcher for {repo}": "openai/gpt-4o"}):
        agent = _agent("Researcher for {repo}")
    mapped = agent.llm
    assert mapped.model == "gpt-4o"

    agent.interpolate_inputs({"repo": "crewAIInc/x"})

    assert agent.role == "Researcher for crewAIInc/x"
    assert agent.llm is mapped


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


def test_the_streaming_flag_of_the_replaced_instance_follows_swap_and_revert() -> None:
    """`Crew.kickoff(stream=True)` flags the instance the agent runs on at that
    moment; whatever the interpolation then puts in its place must carry it."""
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = _agent("Researcher for {repo}")
        declared = agent.llm

        # Kickoff 1 does not stream.
        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o" and agent.llm.stream is False

        # Kickoff 2 streams — the flag lands on the gpt-4o instance — and its
        # input interpolates to a role that is not a key.
        agent.llm.stream = True
        agent.interpolate_inputs({"repo": "crewAIInc/y"})
        assert agent.llm is declared and declared.stream is True

        # Kickoff 3 is back on the key.
        agent.interpolate_inputs({"repo": "crewAIInc/x"})
        assert agent.llm.model == "gpt-4o" and agent.llm.stream is True


def test_a_same_provider_swap_keeps_the_declared_configuration() -> None:
    """The mapped model is built like the declared llm, not from a bare string.

    A new instance of the same class, so everything derived from the model is
    computed for the new one; the endpoint, key and generation settings the
    caller put on the declared llm come along, all the way into the SDK client.
    """
    declared = _configured_llm()
    with llm_overlay(TEMPLATE_OVERLAY):
        agent = Agent(
            role="Researcher for {repo}", goal="g", backstory="b", llm=declared
        )
        agent.interpolate_inputs({"repo": "crewAIInc/x"})

    swapped = agent.llm
    assert swapped is not declared and type(swapped) is OpenAICompletion
    assert swapped.model == "gpt-4o" and declared.model == "gpt-4o-mini"
    assert _configuration_of(swapped) == CONFIGURATION
    assert str(swapped._client.base_url) == "http://localhost:9999/v1/"
    assert swapped._client.timeout == 42


def test_a_cross_provider_swap_carries_settings_but_not_credentials() -> None:
    """Generation settings mean the same thing everywhere; an endpoint and a
    key belong to the provider they were issued for. The Anthropic instance
    gets Anthropic's own defaults and environment for those."""
    declared = _configured_llm()
    with llm_overlay({"Researcher": "anthropic/claude-haiku-4-5"}):
        agent = Agent(role="Researcher", goal="g", backstory="b", llm=declared)

    swapped = agent.llm
    assert isinstance(swapped, AnthropicCompletion)
    assert swapped.model == "claude-haiku-4-5"
    assert swapped.timeout == 42 and swapped.temperature == 0.1
    assert swapped.max_tokens == 77
    assert swapped.api_key != "k" and swapped.base_url is None
    assert swapped.additional_params == {}


def test_a_setting_the_provider_derived_from_the_model_is_not_carried() -> None:
    """Anthropic fills `max_tokens` with the model's output cap when the caller
    did not set it; pinning one model's cap on another is a 400 waiting to
    happen. The new model derives its own — a cap the caller did set is kept."""
    with llm_overlay({"Researcher": "anthropic/claude-haiku-4-5"}):
        derived = Agent(
            role="Researcher",
            goal="g",
            backstory="b",
            llm="anthropic/claude-sonnet-4-6",
        )
        explicit = Agent(
            role="Researcher",
            goal="g",
            backstory="b",
            llm=LLM(model="anthropic/claude-sonnet-4-6", max_tokens=500),
        )
    haiku = LLM(model="anthropic/claude-haiku-4-5")
    sonnet = LLM(model="anthropic/claude-sonnet-4-6")
    assert sonnet.max_tokens != haiku.max_tokens

    assert derived.llm.model == "claude-haiku-4-5"
    assert derived.llm.max_tokens == haiku.max_tokens
    assert explicit.llm.max_tokens == 500


@pytest.mark.filterwarnings("ignore:LiteAgent is deprecated")
def test_the_construction_time_overlay_keeps_the_declared_configuration_too() -> None:
    """The #7500 path — the agent is built inside the block with its role a key —
    goes through the same construction as the kickoff-time swap."""
    with llm_overlay(OVERLAY):
        agent = Agent(role="Researcher", goal="g", backstory="b", llm=_configured_llm())
        lite = LiteAgent(
            role="Researcher", goal="g", backstory="b", llm=_configured_llm()
        )

    for built in (agent.llm, lite.llm):
        assert type(built) is OpenAICompletion and built.model == "gpt-4o"
        assert _configuration_of(built) == CONFIGURATION


def test_a_crew_copy_made_inside_the_block_is_built_from_the_declared_llm() -> None:
    """`Crew.copy()` — the `kickoff_for_each` path — copies every agent before its
    input is interpolated. A template that is itself a key maps at construction;
    a copy built from that mapped llm would record it as its declared one and
    never revert, so the copy is built from the declared llm and resolves the
    overlay for its own role."""
    with llm_overlay({"Researcher for {repo}": "openai/gpt-4o"}):
        agent = _agent("Researcher for {repo}")
        task = Task(description="d", expected_output="e", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        assert agent.llm.model == "gpt-4o"

        copy = crew.copy().agents[0]
        assert copy is not agent and copy.llm.model == "gpt-4o"

        copy.interpolate_inputs({"repo": "crewAIInc/x"})
        assert copy.role == "Researcher for crewAIInc/x"
        assert copy.llm.model == "gpt-4o-mini"

    # Outside any block a copy keeps the llm the agent runs on.
    with llm_overlay(OVERLAY):
        mapped = _agent("Researcher")
    assert mapped.copy().llm.model == "gpt-4o"


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

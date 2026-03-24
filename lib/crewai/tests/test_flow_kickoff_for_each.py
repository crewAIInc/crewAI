"""Regression tests for issue #4555: Flows do not work with kickoff_for_each.

The root cause was that FlowTrackable (pre-1.10) had a ``parent_flow``
Pydantic field typed as ``InstanceOf[Flow[Any]]``.  ``Flow[Any]`` creates a
dynamic ``_FlowGeneric`` subclass via ``__class_getitem__``, so Pydantic's
``isinstance`` check rejected concrete Flow subclasses during ``crew.copy()``.

The fix adds ``"parent_flow"`` to the exclude set in ``Crew.copy()`` so
that even when the field is present (older FlowTrackable or forward-compat),
it is never round-tripped through the Crew constructor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crew() -> Crew:
    agent = Agent(
        role="{topic} Researcher",
        goal="Research {topic}.",
        backstory="Expert on {topic}.",
    )
    task = Task(
        description="Write about {topic}.",
        expected_output="A short paragraph about {topic}.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_copy_excludes_parent_flow_field():
    """crew.copy() must not pass ``parent_flow`` to the new Crew constructor."""
    crew = _make_crew()

    # Simulate the old FlowTrackable behaviour: inject parent_flow via
    # model_dump by monkey-patching the method.
    original_model_dump = crew.model_dump

    def patched_model_dump(self_or_kw=None, **kwargs):
        # Handle both bound call (self, **kw) and unbound call (**kw)
        if self_or_kw is not None and not isinstance(self_or_kw, dict):
            data = original_model_dump(**kwargs)
        else:
            if self_or_kw is not None:
                kwargs.update(self_or_kw)
            data = original_model_dump(**kwargs)
        # Pretend parent_flow leaked through model_dump (old versions),
        # but respect the exclude set so the test fails without the fix and
        # passes with it (i.e. when crew.copy() excludes "parent_flow").
        exclude = kwargs.get("exclude")
        if not (isinstance(exclude, (set, frozenset)) and "parent_flow" in exclude):
            data["parent_flow"] = MagicMock()
        return data

    with patch.object(type(crew), "model_dump", patched_model_dump):
        # This would raise ValidationError without the fix
        copied = crew.copy()

    assert copied is not crew
    assert len(copied.agents) == len(crew.agents)
    assert len(copied.tasks) == len(crew.tasks)


def test_copy_preserves_agents_and_tasks():
    """crew.copy() must produce independent agent/task copies."""
    crew = _make_crew()
    copied = crew.copy()

    assert copied is not crew
    assert len(copied.agents) == len(crew.agents)
    assert len(copied.tasks) == len(crew.tasks)
    # Agents and tasks should be different objects
    assert copied.agents[0] is not crew.agents[0]
    assert copied.tasks[0] is not crew.tasks[0]
    # But same logical content
    assert copied.agents[0].role == crew.agents[0].role
    assert copied.tasks[0].description == crew.tasks[0].description


def test_kickoff_for_each_calls_copy_per_input():
    """kickoff_for_each should call copy() once per input."""
    crew = _make_crew()
    inputs = [{"topic": "dogs"}, {"topic": "cats"}, {"topic": "birds"}]

    copies_made = []
    original_copy = Crew.copy

    def tracking_copy(self_crew):
        c = original_copy(self_crew)
        copies_made.append(c)
        return c

    mock_output = MagicMock(
        raw="output",
        to_dict=MagicMock(return_value={}),
        json_dict=None,
        pydantic=None,
        token_usage={},
    )

    with patch.object(type(crew), "copy", tracking_copy):
        with patch.object(Crew, "kickoff", return_value=mock_output):
            results = crew.kickoff_for_each(inputs=inputs)

    assert len(results) == 3
    assert len(copies_made) == 3
    # Each copy should be a distinct object
    assert len(set(id(c) for c in copies_made)) == 3


def test_copy_with_simulated_parent_flow_attribute():
    """Simulate a crew that has parent_flow set as a regular attribute.

    Even if parent_flow is somehow set on the crew (e.g. by old
    FlowTrackable), copy() should succeed.
    """
    crew = _make_crew()
    # Simulate old FlowTrackable setting parent_flow directly
    object.__setattr__(crew, "parent_flow", MagicMock())

    # copy() should not raise
    copied = crew.copy()
    assert copied is not crew
    assert len(copied.agents) == len(crew.agents)


def test_copy_inside_flow_context():
    """crew.copy() should work when called within a Flow execution context."""
    from crewai.flow.flow_context import current_flow_id, current_flow_request_id

    crew = _make_crew()

    # Simulate being inside a flow execution
    token_id = current_flow_id.set("test-flow-id")
    token_req = current_flow_request_id.set("test-request-id")
    try:
        copied = crew.copy()
        assert copied is not crew
        assert len(copied.agents) == len(crew.agents)
    finally:
        current_flow_id.reset(token_id)
        current_flow_request_id.reset(token_req)

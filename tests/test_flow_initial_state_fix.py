"""Test Flow initial_state BaseModel dict coercion fix for issue #3147"""

from crewai.flow.flow import Flow, FlowState


# All test states now inherit from FlowState to include the mandatory 'id' field
class StateWithItems(FlowState):
    items: list = [1, 2, 3]
    metadata: dict = {"x": 1}


class StateWithKeys(FlowState):
    keys: list = ["a", "b", "c"]
    data: str = "test"


class StateWithValues(FlowState):
    values: list = [10, 20, 30]
    name: str = "example"


class StateWithGet(FlowState):
    get: str = "method_name"
    config: dict = {"enabled": True}


def test_flow_initial_state_items_field() -> None:
    """Test that BaseModel with 'items' field preserves structure and doesn't get dict coercion."""
    flow = Flow(initial_state=StateWithItems())
    flow.kickoff()

    assert isinstance(flow.state, StateWithItems)
    assert not isinstance(flow.state, dict)
    assert isinstance(flow.state.items, list)
    assert flow.state.items == [1, 2, 3]
    assert len(flow.state.items) == 3
    assert flow.state.metadata == {"x": 1}


def test_flow_initial_state_keys_field():
    """Test that BaseModel with 'keys' field preserves structure."""
    flow = Flow(initial_state=StateWithKeys())
    flow.kickoff()

    assert isinstance(flow.state, StateWithKeys)
    assert isinstance(flow.state.keys, list)
    assert flow.state.keys == ["a", "b", "c"]


def test_flow_initial_state_values_field():
    """Test that BaseModel with 'values' field preserves structure."""
    flow = Flow(initial_state=StateWithValues())
    flow.kickoff()

    assert isinstance(flow.state, StateWithValues)
    assert isinstance(flow.state.values, list)
    assert flow.state.values == [10, 20, 30]


def test_flow_initial_state_get_field():
    """Test that BaseModel with 'get' field preserves structure."""
    flow = Flow(initial_state=StateWithGet())
    flow.kickoff()

    assert isinstance(flow.state, StateWithGet)
    assert isinstance(flow.state.get, str)
    assert flow.state.get == "method_name"


def test_flow_with_inputs_preserves_basemodel():
    """Test that providing inputs to flow preserves BaseModel structure."""

    class InputState(FlowState):
        items: list = []
        name: str = ""

    flow = Flow(initial_state=InputState())
    flow.kickoff(inputs={"name": "test_flow", "items": [5, 6, 7]})

    assert isinstance(flow.state, InputState)
    assert not isinstance(flow.state, dict)
    assert flow.state.name == "test_flow"
    assert flow.state.items == [5, 6, 7]


def test_reproduction_case_from_issue_3147():
    """Test the exact reproduction case from GitHub issue #3147."""

    class MyState(FlowState):
        items: list = [1, 2, 3]
        metadata: dict = {"x": 1}

    flow = Flow(initial_state=MyState())
    flow.kickoff()

    assert isinstance(flow.state.items, list)
    assert len(flow.state.items) == 3
    assert flow.state.items == [1, 2, 3]
    assert not callable(flow.state.items)

"""Test Flow initial_state BaseModel dict coercion fix for issue #3147"""

from pydantic import BaseModel

from crewai.flow.flow import Flow


class StateWithItems(BaseModel):
    items: list = [1, 2, 3]
    metadata: dict = {"x": 1}


class StateWithKeys(BaseModel):
    keys: list = ["a", "b", "c"]
    data: str = "test"


class StateWithValues(BaseModel):
    values: list = [10, 20, 30]
    name: str = "example"


class StateWithGet(BaseModel):
    get: str = "method_name"
    config: dict = {"enabled": True}


class StateWithPop(BaseModel):
    pop: int = 42
    settings: list = ["option1", "option2"]


class StateWithUpdate(BaseModel):
    update: bool = True
    version: str = "1.0.0"


class StateWithClear(BaseModel):
    clear: str = "action"
    status: str = "active"


def test_flow_initial_state_items_field():
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
    assert len(flow.state.keys) == 3
    assert flow.state.data == "test"


def test_flow_initial_state_values_field():
    """Test that BaseModel with 'values' field preserves structure."""
    flow = Flow(initial_state=StateWithValues())
    flow.kickoff()
    
    assert isinstance(flow.state, StateWithValues)
    assert isinstance(flow.state.values, list)
    assert flow.state.values == [10, 20, 30]
    assert len(flow.state.values) == 3
    assert flow.state.name == "example"


def test_flow_initial_state_get_field():
    """Test that BaseModel with 'get' field preserves structure."""
    flow = Flow(initial_state=StateWithGet())
    flow.kickoff()
    
    assert isinstance(flow.state, StateWithGet)
    assert isinstance(flow.state.get, str)
    assert flow.state.get == "method_name"
    assert flow.state.config == {"enabled": True}


def test_flow_initial_state_pop_field():
    """Test that BaseModel with 'pop' field preserves structure."""
    flow = Flow(initial_state=StateWithPop())
    flow.kickoff()
    
    assert isinstance(flow.state, StateWithPop)
    assert isinstance(flow.state.pop, int)
    assert flow.state.pop == 42
    assert flow.state.settings == ["option1", "option2"]


def test_flow_initial_state_update_field():
    """Test that BaseModel with 'update' field preserves structure."""
    flow = Flow(initial_state=StateWithUpdate())
    flow.kickoff()
    
    assert isinstance(flow.state, StateWithUpdate)
    assert isinstance(flow.state.update, bool)
    assert flow.state.update is True
    assert flow.state.version == "1.0.0"


def test_flow_initial_state_clear_field():
    """Test that BaseModel with 'clear' field preserves structure."""
    flow = Flow(initial_state=StateWithClear())
    flow.kickoff()
    
    assert isinstance(flow.state, StateWithClear)
    assert isinstance(flow.state.clear, str)
    assert flow.state.clear == "action"
    assert flow.state.status == "active"


def test_flow_state_modification_preserves_basemodel():
    """Test that modifying flow state preserves BaseModel structure."""
    
    class ModifiableState(BaseModel):
        items: list = [1, 2, 3]
        counter: int = 0
    
    class TestFlow(Flow[ModifiableState]):
        @Flow.start()
        def modify_state(self):
            self.state.counter += 1
            self.state.items.append(4)
    
    flow = TestFlow(initial_state=ModifiableState())
    flow.kickoff()
    
    assert isinstance(flow.state, ModifiableState)
    assert not isinstance(flow.state, dict)
    
    assert flow.state.counter == 1
    assert flow.state.items == [1, 2, 3, 4]


def test_flow_with_inputs_preserves_basemodel():
    """Test that providing inputs to flow preserves BaseModel structure."""
    
    class InputState(BaseModel):
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
    
    class MyState(BaseModel):
        items: list = [1, 2, 3]
        metadata: dict = {"x": 1}
    
    flow = Flow(initial_state=MyState())
    flow.kickoff()
    
    assert isinstance(flow.state.items, list)
    assert len(flow.state.items) == 3
    assert flow.state.items == [1, 2, 3]
    
    assert not callable(flow.state.items)
    assert str(type(flow.state.items)) != "<class 'builtin_function_or_method'>"

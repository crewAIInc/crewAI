from typing import Any, List

import litellm
import pytest

from crewai.llm import LLM


class CustomCallback:
    """A simple callback class for testing."""
    pass


class DifferentCallback:
    """A different callback class for testing type differentiation."""
    pass


@pytest.fixture
def reset_litellm_callbacks():
    """Fixture to reset litellm callbacks after each test."""
    original_success_callback = litellm.success_callback
    original_async_success_callback = litellm._async_success_callback
    
    yield
    
    litellm.success_callback = original_success_callback
    litellm._async_success_callback = original_async_success_callback


def test_set_callbacks_handles_removed_callbacks(reset_litellm_callbacks):
    """Test that set_callbacks handles the case where callbacks are removed during iteration."""
    litellm.success_callback = []
    litellm._async_success_callback = []
    
    llm = LLM(model="test-model")
    
    callback1 = CustomCallback()
    callback2 = CustomCallback()
    litellm.success_callback.append(callback1)
    litellm.success_callback.append(callback2)
    
    new_callback = CustomCallback()
    
    litellm.success_callback.remove(callback1)
    
    llm.set_callbacks([new_callback])
    
    assert litellm.callbacks == [new_callback]
    assert len([cb for cb in litellm.success_callback if isinstance(cb, CustomCallback)]) == 0


@pytest.mark.parametrize("callback_count", [1, 3, 5])
def test_set_callbacks_with_different_sizes(callback_count, reset_litellm_callbacks):
    """Test with various numbers of callbacks."""
    litellm.success_callback = []
    litellm._async_success_callback = []
    
    llm = LLM(model="test-model")
    
    callbacks = [CustomCallback() for _ in range(callback_count)]
    for callback in callbacks:
        litellm.success_callback.append(callback)
    
    new_callback = CustomCallback()
    
    llm.set_callbacks([new_callback])
    
    assert litellm.callbacks == [new_callback]
    assert len([cb for cb in litellm.success_callback if isinstance(cb, CustomCallback)]) == 0


def test_set_callbacks_with_different_types(reset_litellm_callbacks):
    """Test that callbacks of different types are handled correctly."""
    litellm.success_callback = []
    litellm._async_success_callback = []
    
    llm = LLM(model="test-model")
    
    custom_callback = CustomCallback()
    different_callback = DifferentCallback()
    
    litellm.success_callback.append(custom_callback)
    litellm.success_callback.append(different_callback)
    
    llm.set_callbacks([CustomCallback()])
    
    assert any(isinstance(cb, DifferentCallback) for cb in litellm.success_callback)
    assert not any(isinstance(cb, CustomCallback) for cb in litellm.success_callback)


def test_set_callbacks_with_empty_list(reset_litellm_callbacks):
    """Test setting callbacks with an empty list."""
    litellm.success_callback = []
    litellm._async_success_callback = []
    
    llm = LLM(model="test-model")
    
    custom_callback = CustomCallback()
    litellm.success_callback.append(custom_callback)
    
    llm.set_callbacks([])
    
    assert litellm.callbacks == []
    assert custom_callback in litellm.success_callback

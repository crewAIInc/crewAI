import litellm
import pytest
from typing import Any

from crewai.llm import LLM


def test_set_callbacks_handles_removed_callbacks():
    """Test that set_callbacks handles the case where callbacks are removed during iteration."""
    class CustomCallback:
        pass

    original_success_callback = litellm.success_callback
    original_async_success_callback = litellm._async_success_callback
    
    try:
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
    
    finally:
        litellm.success_callback = original_success_callback
        litellm._async_success_callback = original_async_success_callback

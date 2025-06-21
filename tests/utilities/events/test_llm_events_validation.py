import pytest
from crewai.utilities.events.llm_events import LLMCallStartedEvent
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TestLLMCallStartedEventValidation:
    """Test cases for LLMCallStartedEvent validation and sanitization"""

    def test_normal_dict_tools_work(self):
        """Test that normal dict tools work correctly"""
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[{"name": "tool1"}, {"name": "tool2"}],
            callbacks=None
        )
        assert event.tools == [{"name": "tool1"}, {"name": "tool2"}]
        assert event.type == "llm_call_started"

    def test_token_calc_handler_in_tools_filtered_out(self):
        """Test that TokenCalcHandler objects in tools list are filtered out"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[{"name": "tool1"}, token_handler, {"name": "tool2"}],
            callbacks=None
        )
        
        assert event.tools == [{"name": "tool1"}, {"name": "tool2"}]
        assert len(event.tools) == 2

    def test_mixed_objects_in_tools_only_dicts_preserved(self):
        """Test that only dict objects are preserved when mixed types are in tools"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[
                {"name": "tool1"},
                token_handler,
                "string_tool",
                {"name": "tool2"},
                123,
                {"name": "tool3"}
            ],
            callbacks=None
        )
        
        assert event.tools == [{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
        assert len(event.tools) == 3

    def test_empty_tools_list_handled(self):
        """Test that empty tools list is handled correctly"""
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[],
            callbacks=None
        )
        assert event.tools == []

    def test_none_tools_handled(self):
        """Test that None tools value is handled correctly"""
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=None,
            callbacks=None
        )
        assert event.tools is None

    def test_all_non_dict_tools_results_in_empty_list(self):
        """Test that when all tools are non-dict objects, result is empty list"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[token_handler, "string_tool", 123, ["list_tool"]],
            callbacks=None
        )
        
        assert event.tools == []

    def test_reproduction_case_from_issue_3043(self):
        """Test the exact reproduction case from GitHub issue #3043"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[{"name": "tool1"}, token_handler],
            callbacks=None
        )
        
        assert event.tools == [{"name": "tool1"}]
        assert len(event.tools) == 1

    def test_callbacks_with_token_handler_still_work(self):
        """Test that TokenCalcHandler in callbacks still works normally"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[{"name": "tool1"}],
            callbacks=[token_handler]
        )
        
        assert event.tools == [{"name": "tool1"}]
        assert event.callbacks == [token_handler]

    def test_string_messages_work(self):
        """Test that string messages work with tool sanitization"""
        token_handler = TokenCalcHandler(TokenProcess())
        
        event = LLMCallStartedEvent(
            messages="test message",
            tools=[{"name": "tool1"}, token_handler],
            callbacks=None
        )
        
        assert event.messages == "test message"
        assert event.tools == [{"name": "tool1"}]

    def test_available_functions_preserved(self):
        """Test that available_functions are preserved during sanitization"""
        token_handler = TokenCalcHandler(TokenProcess())
        available_funcs = {"func1": lambda x: x}
        
        event = LLMCallStartedEvent(
            messages=[{"role": "user", "content": "test message"}],
            tools=[{"name": "tool1"}, token_handler],
            callbacks=None,
            available_functions=available_funcs
        )
        
        assert event.tools == [{"name": "tool1"}]
        assert event.available_functions == available_funcs

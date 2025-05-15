import pytest

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import LLM
from crewai.utilities.token_counter_callback import TokenCalcHandler


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    llm.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()

    llm.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()


class TestLLMStopWords:
    """Tests for LLM stop words functionality."""
    
    def test_supports_stop_words_for_o3_model(self):
        """Test that supports_stop_words returns False for o3 model."""
        llm = LLM(model="o3")
        assert not llm.supports_stop_words()
    
    def test_supports_stop_words_for_o4_mini_model(self):
        """Test that supports_stop_words returns False for o4-mini model."""
        llm = LLM(model="o4-mini")
        assert not llm.supports_stop_words()
    
    def test_supports_stop_words_for_supported_model(self):
        """Test that supports_stop_words returns True for models that support stop words."""
        llm = LLM(model="gpt-4")
        assert llm.supports_stop_words()
    
    @pytest.mark.vcr(filter_headers=["authorization"])
    def test_llm_call_excludes_stop_parameter_for_unsupported_models(self, monkeypatch):
        """Test that the LLM.call method excludes the stop parameter for models that don't support it."""
        def mock_completion(**kwargs):
            assert 'stop' not in kwargs, "Stop parameter should be excluded for o3 model"
            assert 'model' in kwargs, "Model parameter should be included"
            assert 'messages' in kwargs, "Messages parameter should be included"
            return {"choices": [{"message": {"content": "Hello, World!"}}]}
        
        monkeypatch.setattr("litellm.completion", mock_completion)
        
        llm = LLM(model="o3")
        llm.stop = ["STOP"]
        
        messages = [{"role": "user", "content": "Say 'Hello, World!'"}]
        response = llm.call(messages)
        
        assert response == "Hello, World!"

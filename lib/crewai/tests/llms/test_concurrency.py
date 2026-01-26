import threading
import time
from unittest.mock import MagicMock, patch
import pytest
from crewai.llm import LLM
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess

def test_concurrent_llm_calls_isolation():
    """
    Test that concurrent LLM calls with different callbacks do not interfere with each other.
    """
    
    # We patch globally so it applies to all threads
    # We use crewai.llm.litellm to be safe
    with patch("crewai.llm.litellm.completion") as mock_completion:
        
        # Setup mock to return a valid response structure
        def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0]["content"] if messages else ""
            thread_id = content.split("thread ")[-1]
            
            mock_message = MagicMock()
            mock_message.content = f"Response for thread {thread_id}"
            mock_choice = MagicMock()
            mock_choice.message = mock_message
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            
            # Use unique usage stats based on thread ID
            tid_int = int(thread_id)
            
            # Create a usage object with attributes (not a dict)
            usage_obj = MagicMock()
            usage_obj.prompt_tokens = 10 + tid_int
            usage_obj.completion_tokens = 5 + tid_int
            usage_obj.total_tokens = 15 + 2 * tid_int
            # Mock prompt_tokens_details to be None or have cached_tokens=0
            usage_obj.prompt_tokens_details = None
            
            mock_response.usage = usage_obj
            
            # Simulate slight delay
            time.sleep(0.1)
            return mock_response

        mock_completion.side_effect = side_effect

        # Define the workload
        def run_llm_request(thread_id, result_container):
            token_process = TokenProcess()
            handler = TokenCalcHandler(token_cost_process=token_process)
            
            # Store handler so we can verify it later
            result_container[thread_id] = {
                "handler": handler,
                "summary": None
            }
            
            llm = LLM(model="gpt-4o-mini", is_litellm=True)
            
            llm.call(
                messages=[{"role": "user", "content": f"Hello from thread {thread_id}"}],
                callbacks=[handler]
            )
            
            result_container[thread_id]["summary"] = token_process.get_summary()

        results = {}
        threads = []
        
        # Start threads
        for i in [1, 2]:
            t = threading.Thread(target=run_llm_request, args=(i, results))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Verification
        assert mock_completion.call_count == 2
        
        # Check each call arguments
        for call_args in mock_completion.call_args_list:
            kwargs = call_args.kwargs
            messages = kwargs.get("messages", [])
            content = messages[0]["content"]
            thread_id = int(content.split("thread ")[-1])
            
            expected_handler = results[thread_id]["handler"]
            
            # CRITICAL CHECK: Verify ONLY the expected handler was passed
            assert "callbacks" in kwargs
            callbacks = kwargs["callbacks"]
            assert len(callbacks) == 1
            assert callbacks[0] == expected_handler, f"Callback mismatch for thread {thread_id}"
            
        # Verify token usage isolation
        summary1 = results[1]["summary"]
        assert summary1.prompt_tokens == 11
        assert summary1.completion_tokens == 6
        
        summary2 = results[2]["summary"]
        assert summary2.prompt_tokens == 12
        assert summary2.completion_tokens == 7

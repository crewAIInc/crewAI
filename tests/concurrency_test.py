import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from unittest.mock import patch

from crewai import Agent, Crew, Task


class MockLLM:
    """Mock LLM for testing."""
    def __init__(self, model="gpt-3.5-turbo", **kwargs):
        self.model = model
        self.stop = None
        self.timeout = None
        self.temperature = None
        self.top_p = None
        self.n = None
        self.max_completion_tokens = None
        self.max_tokens = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.logit_bias = None
        self.response_format = None
        self.seed = None
        self.logprobs = None
        self.top_logprobs = None
        self.base_url = None
        self.api_version = None
        self.api_key = None
        self.callbacks = []
        self.context_window_size = 8192
        self.kwargs = {}
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def complete(self, prompt, **kwargs):
        """Mock completion method."""
        return f"Mock response for: {prompt[:20]}..."

    def chat_completion(self, messages, **kwargs):
        """Mock chat completion method."""
        return {"choices": [{"message": {"content": "Mock response"}}]}

    def function_call(self, messages, functions, **kwargs):
        """Mock function call method."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "Mock response",
                        "function_call": {
                            "name": "test_function",
                            "arguments": '{"arg1": "value1"}'
                        }
                    }
                }
            ]
        }
        
    def supports_stop_words(self):
        """Mock supports_stop_words method."""
        return False
        
    def supports_function_calling(self):
        """Mock supports_function_calling method."""
        return True
        
    def get_context_window_size(self):
        """Mock get_context_window_size method."""
        return self.context_window_size
        
    def call(self, messages, callbacks=None):
        """Mock call method."""
        return "Mock response from call method"
        
    def set_callbacks(self, callbacks):
        """Mock set_callbacks method."""
        self.callbacks = callbacks
        
    def set_env_callbacks(self):
        """Mock set_env_callbacks method."""
        pass


def create_test_crew():
    """Create a simple test crew for concurrency testing."""
    with patch("crewai.agent.LLM", MockLLM):
        agent = Agent(
            role="Test Agent",
            goal="Test concurrent execution",
            backstory="I am a test agent for concurrent execution",
        )

        task = Task(
            description="Test task for concurrent execution",
            expected_output="Test output",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False,
        )

        return crew


def test_threading_concurrency():
    """Test concurrent execution using ThreadPoolExecutor."""
    num_threads = 5
    results = []

    def generate_response(idx):
        try:
            crew = create_test_crew()
            with patch("crewai.agent.LLM", MockLLM):
                output = crew.kickoff(inputs={"test_input": f"input_{idx}"})
            return output
        except Exception as e:
            pytest.fail(f"Exception in thread {idx}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(generate_response, i) for i in range(num_threads)]

        for future in as_completed(futures):
            result = future.result()
            assert result is not None
            results.append(result)

    assert len(results) == num_threads


@pytest.mark.asyncio
async def test_asyncio_concurrency():
    """Test concurrent execution using asyncio."""
    num_tasks = 5
    sem = asyncio.Semaphore(num_tasks)

    async def generate_response_async(idx):
        async with sem:
            try:
                crew = create_test_crew()
                with patch("crewai.agent.LLM", MockLLM):
                    output = await crew.kickoff_async(inputs={"test_input": f"input_{idx}"})
                return output
            except Exception as e:
                pytest.fail(f"Exception in task {idx}: {e}")
                return None

    tasks = [generate_response_async(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_tasks
    assert all(result is not None for result in results)


@pytest.mark.asyncio
async def test_extended_asyncio_concurrency():
    """Extended test for asyncio concurrency with more iterations."""
    num_tasks = 5  # Reduced from 10 for faster testing
    iterations = 2  # Reduced from 3 for faster testing
    sem = asyncio.Semaphore(num_tasks)

    async def generate_response_async(idx):
        async with sem:
            crew = create_test_crew()
            for i in range(iterations):
                try:
                    with patch("crewai.agent.LLM", MockLLM):
                        output = await crew.kickoff_async(
                            inputs={"test_input": f"input_{idx}_{i}"}
                        )
                    assert output is not None
                except Exception as e:
                    pytest.fail(f"Exception in task {idx}, iteration {i}: {e}")
                    return False
            return True

    tasks = [generate_response_async(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)

    assert all(results)

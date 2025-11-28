"""Tests for async method validation in crew decorators."""

import pytest

from crewai.project import (
    after_kickoff,
    agent,
    before_kickoff,
    cache_handler,
    callback,
    crew,
    llm,
    task,
    tool,
)


class TestAsyncDecoratorValidation:
    """Test that decorators properly reject async methods with clear error messages."""

    def test_agent_decorator_rejects_async_method(self):
        """Test that @agent decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @agent
            async def async_agent(self):
                return None

        assert "@agent decorator does not support async methods" in str(exc_info.value)
        assert "async_agent" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_task_decorator_rejects_async_method(self):
        """Test that @task decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @task
            async def async_task(self):
                return None

        assert "@task decorator does not support async methods" in str(exc_info.value)
        assert "async_task" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_crew_decorator_rejects_async_method(self):
        """Test that @crew decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @crew
            async def async_crew(self):
                return None

        assert "@crew decorator does not support async methods" in str(exc_info.value)
        assert "async_crew" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_llm_decorator_rejects_async_method(self):
        """Test that @llm decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @llm
            async def async_llm(self):
                return None

        assert "@llm decorator does not support async methods" in str(exc_info.value)
        assert "async_llm" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_tool_decorator_rejects_async_method(self):
        """Test that @tool decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @tool
            async def async_tool(self):
                return None

        assert "@tool decorator does not support async methods" in str(exc_info.value)
        assert "async_tool" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_callback_decorator_rejects_async_method(self):
        """Test that @callback decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @callback
            async def async_callback(self):
                return None

        assert "@callback decorator does not support async methods" in str(exc_info.value)
        assert "async_callback" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_cache_handler_decorator_rejects_async_method(self):
        """Test that @cache_handler decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @cache_handler
            async def async_cache_handler(self):
                return None

        assert "@cache_handler decorator does not support async methods" in str(
            exc_info.value
        )
        assert "async_cache_handler" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_before_kickoff_decorator_rejects_async_method(self):
        """Test that @before_kickoff decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @before_kickoff
            async def async_before_kickoff(self, inputs):
                return inputs

        assert "@before_kickoff decorator does not support async methods" in str(
            exc_info.value
        )
        assert "async_before_kickoff" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_after_kickoff_decorator_rejects_async_method(self):
        """Test that @after_kickoff decorator raises TypeError for async methods."""
        with pytest.raises(TypeError) as exc_info:

            @after_kickoff
            async def async_after_kickoff(self, outputs):
                return outputs

        assert "@after_kickoff decorator does not support async methods" in str(
            exc_info.value
        )
        assert "async_after_kickoff" in str(exc_info.value)
        assert "synchronous method" in str(exc_info.value)

    def test_sync_methods_still_work(self):
        """Test that synchronous methods are still properly decorated."""
        from crewai import Agent, Task

        @agent
        def sync_agent(self):
            return Agent(
                role="Test Agent", goal="Test Goal", backstory="Test Backstory"
            )

        @task
        def sync_task(self):
            return Task(description="Test Description", expected_output="Test Output")

        class TestCrew:
            pass

        test_instance = TestCrew()
        agent_result = sync_agent(test_instance)
        task_result = sync_task(test_instance)

        assert agent_result.role == "Test Agent"
        assert task_result.description == "Test Description"

    def test_error_message_includes_workaround_suggestions(self):
        """Test that error messages include helpful workaround suggestions."""
        with pytest.raises(TypeError) as exc_info:

            @agent
            async def async_agent_with_tools(self):
                return None

        error_message = str(exc_info.value)
        assert "Creating tools/resources synchronously" in error_message
        assert "asyncio.run()" in error_message

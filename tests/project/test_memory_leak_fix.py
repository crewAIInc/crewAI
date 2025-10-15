"""Tests for memory leak fixes in CrewBase decorator."""

import gc
import weakref
from typing import ClassVar
from unittest.mock import MagicMock

from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.project.utils import memoize


class TestMemoizeMemoryLeak:
    """Test that the memoize decorator doesn't cause memory leaks."""

    def test_memoize_allows_garbage_collection(self):
        """Test that memoized methods don't prevent garbage collection."""

        class TestClass:
            def __init__(self, value):
                self.value = value

            @memoize
            def get_value(self):
                return self.value

        # Create instance and get weak reference
        instance = TestClass("test")
        weak_ref = weakref.ref(instance)

        # Call memoized method to populate cache
        result = instance.get_value()
        assert result == "test"

        # Delete the instance
        del instance

        # Force garbage collection
        gc.collect()

        # The weak reference should be None (object was garbage collected)
        assert weak_ref() is None

    def test_memoize_cache_functionality(self):
        """Test that memoization still works correctly."""
        call_count = 0

        class TestClass:
            @memoize
            def expensive_operation(self, x):
                nonlocal call_count
                call_count += 1
                return x * 2

        instance = TestClass()

        # First call
        result1 = instance.expensive_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument should use cache
        result2 = instance.expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Different argument should call function again
        result3 = instance.expensive_operation(10)
        assert result3 == 20
        assert call_count == 2

    def test_memoize_clear_cache(self):
        """Test that cache can be cleared."""
        call_count = 0

        class TestClass:
            @memoize
            def get_data(self):
                nonlocal call_count
                call_count += 1
                return "data"

        instance = TestClass()

        # Call method
        result1 = instance.get_data()
        assert result1 == "data"
        assert call_count == 1

        # Call again, should use cache
        result2 = instance.get_data()
        assert result2 == "data"
        assert call_count == 1

        # Clear cache
        instance.get_data.clear_cache()

        # Call again, should execute function
        result3 = instance.get_data()
        assert result3 == "data"
        assert call_count == 2


class TestCrewBaseMemoryLeak:
    """Test that CrewBase instances can be garbage collected."""

    def test_crewbase_garbage_collection(self):
        """Test that CrewBase instances can be garbage collected."""

        def inner_test():
            @CrewBase
            class TestCrew:
                @agent
                def test_agent(self) -> Agent:
                    return Agent(
                        role="Test Agent", goal="Test goal", backstory="Test backstory"
                    )

                @task
                def test_task(self) -> Task:
                    return Task(
                        description="Test task",
                        expected_output="Test output",
                        agent=self.test_agent(),
                    )

                @crew
                def crew(self) -> Crew:
                    return Crew(agents=[self.test_agent()], tasks=[self.test_task()])

            # Create instance and weak reference
            crew_instance = TestCrew()
            weak_ref = weakref.ref(crew_instance)

            # Use the crew to populate caches
            crew_obj = crew_instance.crew()
            assert crew_obj is not None

            # Delete references
            del crew_obj
            del crew_instance

            return weak_ref

        # Run test in isolated scope
        weak_ref = inner_test()

        # Force garbage collection
        gc.collect()

        # The weak reference should be None (object was garbage collected)
        assert weak_ref() is None

    def test_crewbase_explicit_cleanup(self):
        """Test that explicit cleanup works."""

        def inner_test():
            @CrewBase
            class TestCrew:
                @agent
                def test_agent(self) -> Agent:
                    return Agent(
                        role="Test Agent", goal="Test goal", backstory="Test backstory"
                    )

                @task
                def test_task(self) -> Task:
                    return Task(
                        description="Test task",
                        expected_output="Test output",
                        agent=self.test_agent(),
                    )

                @crew
                def crew(self) -> Crew:
                    return Crew(agents=[self.test_agent()], tasks=[self.test_task()])

            # Create instance
            crew_instance = TestCrew()
            weak_ref = weakref.ref(crew_instance)

            # Use the crew
            crew_obj = crew_instance.crew()
            assert crew_obj is not None

            # Explicit cleanup
            crew_instance.cleanup()

            # Delete references
            del crew_obj
            del crew_instance

            return weak_ref

        # Run test in isolated scope
        weak_ref = inner_test()

        # Force garbage collection
        gc.collect()

        # Should be garbage collected
        assert weak_ref() is None

    def test_multiple_crewbase_instances(self):
        """Test that multiple CrewBase instances don't interfere with each other's garbage collection."""

        def inner_test():
            @CrewBase
            class TestCrew:
                def __init__(self, name):
                    super().__init__()
                    self.name = name

                @agent
                def test_agent(self) -> Agent:
                    return Agent(
                        role=f"Test Agent {self.name}",
                        goal="Test goal",
                        backstory="Test backstory",
                    )

                @task
                def test_task(self) -> Task:
                    return Task(
                        description=f"Test task for {self.name}",
                        expected_output="Test output",
                        agent=self.test_agent(),
                    )

                @crew
                def crew(self) -> Crew:
                    return Crew(agents=[self.test_agent()], tasks=[self.test_task()])

            # Create multiple instances
            instances = []
            weak_refs = []

            for i in range(5):
                instance = TestCrew(f"crew_{i}")
                instances.append(instance)
                weak_refs.append(weakref.ref(instance))

                # Use each crew
                crew_obj = instance.crew()
                assert crew_obj is not None
                del crew_obj  # Clean up immediately

            # Delete all instances
            del instances

            return weak_refs

        # Run test in isolated scope
        weak_refs = inner_test()

        # Force garbage collection
        gc.collect()

        # All instances should be garbage collected
        for weak_ref in weak_refs:
            assert weak_ref() is None

    def test_crewbase_with_mcp_adapter_cleanup(self):
        """Test that MCP adapter is properly cleaned up."""

        @CrewBase
        class TestCrewWithMCP:
            mcp_server_params: ClassVar[dict] = {"test": "params"}

            @agent
            def test_agent(self) -> Agent:
                return Agent(
                    role="Test Agent", goal="Test goal", backstory="Test backstory"
                )

            @crew
            def crew(self) -> Crew:
                return Crew(agents=[self.test_agent()], tasks=[])

        # Create instance
        crew_instance = TestCrewWithMCP()

        # Mock MCP adapter
        mock_adapter = MagicMock()
        crew_instance._mcp_server_adapter = mock_adapter

        # Create weak reference
        weak_ref = weakref.ref(crew_instance)

        # Cleanup
        crew_instance.cleanup()

        # Verify MCP adapter stop was called
        mock_adapter.stop.assert_called_once()

        # Delete instance
        del crew_instance

        # Force garbage collection
        gc.collect()

        # Should be garbage collected
        assert weak_ref() is None


class TestMemoryLeakRegression:
    """Regression tests to ensure the original issue is fixed."""

    def test_issue_3450_regression(self):
        """
        Regression test for issue #3450.

        This test simulates the exact scenario from the GitHub issue
        to ensure the memory leak is fixed.
        """

        def inner_test():
            @CrewBase
            class CrewAI:
                @agent
                def searcher(self) -> Agent:
                    return Agent(
                        role="Searcher",
                        goal="Search for information",
                        backstory="I am a search agent",
                    )

                @task
                def searcher_task(self) -> Task:
                    return Task(
                        description="Search for information",
                        expected_output="Search results",
                        agent=self.searcher(),
                    )

                @crew
                def crew(self) -> Crew:
                    return Crew(
                        agents=[self.searcher()],
                        tasks=[self.searcher_task()],
                        verbose=False,
                        cache=True,
                        memory=False,
                    )

            # Create multiple instances to simulate the issue
            instances = []
            weak_refs = []

            for _ in range(10):
                instance = CrewAI()
                instances.append(instance)
                weak_refs.append(weakref.ref(instance))

                # Execute crew to populate caches
                crew_obj = instance.crew()
                assert crew_obj is not None
                del crew_obj  # Clean up immediately

            # Clear all references
            del instances

            return weak_refs

        # Run test in isolated scope
        weak_refs = inner_test()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Count how many instances are still alive
        alive_count = sum(1 for ref in weak_refs if ref() is not None)

        # All instances should be garbage collected
        assert alive_count == 0, f"{alive_count} instances were not garbage collected"

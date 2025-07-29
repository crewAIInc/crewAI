#!/usr/bin/env python3
"""
Tests for Flow-level RPM control functionality.
"""

import gc
import pytest
from unittest.mock import Mock, patch

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start, listen
from crewai.utilities.rpm_controller import RPMController


class TestFlowRPMControl:
    """Test suite for Flow-level RPM control."""

    def test_flow_initialization_with_rpm(self):
        """Test that Flow initializes correctly with max_rpm parameter."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Test with RPM limit
        flow_with_rpm = TestFlow(max_rpm=15)
        assert flow_with_rpm.max_rpm == 15
        assert flow_with_rpm._rpm_controller is not None
        assert flow_with_rpm._rpm_controller.max_rpm == 15

        # Test without RPM limit
        flow_without_rpm = TestFlow()
        assert flow_without_rpm.max_rpm is None
        assert flow_without_rpm._rpm_controller is None

    def test_flow_initialization_invalid_rpm(self):
        """Test Flow initialization with invalid RPM values."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Test negative RPM
        with pytest.raises(ValueError, match="max_rpm must be a positive integer"):
            TestFlow(max_rpm=-1)

        # Test zero RPM
        with pytest.raises(ValueError, match="max_rpm must be a positive integer"):
            TestFlow(max_rpm=0)

        # Test non-integer RPM
        with pytest.raises(ValueError, match="max_rpm must be a positive integer"):
            TestFlow(max_rpm=10.5)

        # Test string RPM
        with pytest.raises(ValueError, match="max_rpm must be a positive integer"):
            TestFlow(max_rpm="10")

    def test_flow_initialization_with_verbose(self):
        """Test that Flow initializes correctly with verbose parameter."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        flow_verbose = TestFlow(max_rpm=10, verbose=True)
        assert flow_verbose._logger.verbose is True

        flow_not_verbose = TestFlow(max_rpm=10, verbose=False)
        assert flow_not_verbose._logger.verbose is False

    def test_get_flow_rpm_controller(self):
        """Test the get_flow_rpm_controller method."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Flow with RPM
        flow_with_rpm = TestFlow(max_rpm=20)
        controller = flow_with_rpm.get_flow_rpm_controller()
        assert controller is not None
        assert isinstance(controller, RPMController)
        assert controller.max_rpm == 20

        # Flow without RPM
        flow_without_rpm = TestFlow()
        controller = flow_without_rpm.get_flow_rpm_controller()
        assert controller is None

    def test_crew_rpm_override_by_flow(self):
        """Test that crew's RPM settings are overridden by Flow's global RPM."""

        class TestFlow(Flow):
            def __init__(self):
                super().__init__(max_rpm=5)

            @start()
            def create_crew(self):
                agent = Agent(
                    role="Test Agent",
                    goal="Test goal",
                    backstory="Test backstory"
                )

                task = Task(
                    description="Test task",
                    agent=agent,
                    expected_output="Test output"
                )

                # Create crew with different RPM limit
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    max_rpm=30  # This should be overridden
                )

                return crew

        flow = TestFlow()
        # This would normally execute the flow, but for testing we'll mock the execution
        # to avoid actually running the crew
        with patch.object(flow, '_execute_method') as mock_execute:
            # Mock the method to return a crew directly
            agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
            task = Task(description="Test task", agent=agent, expected_output="Test output")
            mock_crew = Crew(agents=[agent], tasks=[task], max_rpm=30)
            mock_execute.return_value = mock_crew

            # Configure the crew with flow's RPM controller
            flow.set_crew_rpm_controller(mock_crew)

            # Verify that the crew now uses the flow's RPM controller
            assert mock_crew._rpm_controller is flow._rpm_controller
            assert mock_crew._rpm_controller.max_rpm == 5

    def test_set_crew_rpm_controller(self):
        """Test the set_crew_rpm_controller method."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Create flow with RPM controller
        flow = TestFlow(max_rpm=12)

        # Create a crew
        agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
        task = Task(description="Test task", agent=agent, expected_output="Test output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=25)

        # Store original controllers for comparison
        original_crew_controller = crew._rpm_controller
        original_agent_controller = agent._rpm_controller

        # Apply flow's RPM controller to crew
        flow.set_crew_rpm_controller(crew)

        # Verify that crew and its agents now use flow's controller
        assert crew._rpm_controller is flow._rpm_controller
        assert crew._rpm_controller is not original_crew_controller
        assert agent._rpm_controller is flow._rpm_controller
        assert agent._rpm_controller is not original_agent_controller

    def test_set_crew_rpm_controller_no_flow_rpm(self):
        """Test set_crew_rpm_controller when flow has no RPM limit."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Create flow without RPM controller
        flow = TestFlow()

        # Create a crew
        agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
        task = Task(description="Test task", agent=agent, expected_output="Test output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=25)

        # Store original controllers
        original_crew_controller = crew._rpm_controller

        # Try to apply flow's RPM controller (should do nothing)
        flow.set_crew_rpm_controller(crew)

        # Verify that crew keeps its original controller
        assert crew._rpm_controller is original_crew_controller

    def test_auto_configure_flow_crews_function(self):
        """Test the auto_configure_flow_crews utility function."""
        from crewai.flow.utils import auto_configure_flow_crews

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        # Create flow with RPM controller
        flow = TestFlow(max_rpm=8)

        # Create test crew
        agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
        task = Task(description="Test task", agent=agent, expected_output="Test output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=20)

        # Store original controller
        original_controller = crew._rpm_controller

        # Auto-configure crew
        result = auto_configure_flow_crews(flow, crew)

        # Verify crew was configured and returned
        assert result is crew
        assert crew._rpm_controller is flow._rpm_controller
        assert crew._rpm_controller is not original_controller

    def test_auto_configure_flow_crews_with_list(self):
        """Test auto_configure_flow_crews with a list containing crews."""
        from crewai.flow.utils import auto_configure_flow_crews

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        flow = TestFlow(max_rpm=6)

        # Create test crews
        agent1 = Agent(role="Agent 1", goal="Goal 1", backstory="Backstory 1")
        task1 = Task(description="Task 1", agent=agent1, expected_output="Output 1")
        crew1 = Crew(agents=[agent1], tasks=[task1], max_rpm=15)

        agent2 = Agent(role="Agent 2", goal="Goal 2", backstory="Backstory 2")
        task2 = Task(description="Task 2", agent=agent2, expected_output="Output 2")
        crew2 = Crew(agents=[agent2], tasks=[task2], max_rpm=25)

        test_list = [crew1, "not_a_crew", crew2]

        # Auto-configure
        result = auto_configure_flow_crews(flow, test_list)

        # Verify both crews were configured
        assert result is test_list
        assert crew1._rpm_controller is flow._rpm_controller
        assert crew2._rpm_controller is flow._rpm_controller

    def test_auto_configure_flow_crews_with_dict(self):
        """Test auto_configure_flow_crews with a dictionary containing crews."""
        from crewai.flow.utils import auto_configure_flow_crews

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        flow = TestFlow(max_rpm=4)

        # Create test crew
        agent = Agent(role="Agent", goal="Goal", backstory="Backstory")
        task = Task(description="Task", agent=agent, expected_output="Output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=18)

        test_dict = {
            "crew": crew,
            "other": "not_a_crew"
        }

        # Auto-configure
        result = auto_configure_flow_crews(flow, test_dict)

        # Verify crew was configured
        assert result is test_dict
        assert crew._rpm_controller is flow._rpm_controller

    def test_auto_configure_flow_crews_no_flow_rpm(self):
        """Test auto_configure_flow_crews when flow has no RPM limit."""
        from crewai.flow.utils import auto_configure_flow_crews

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        flow = TestFlow()  # No RPM limit

        # Create test crew
        agent = Agent(role="Agent", goal="Goal", backstory="Backstory")
        task = Task(description="Task", agent=agent, expected_output="Output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=12)

        original_controller = crew._rpm_controller

        # Auto-configure (should do nothing)
        result = auto_configure_flow_crews(flow, crew)

        # Verify crew was not modified
        assert result is crew
        assert crew._rpm_controller is original_controller

    def test_crew_set_flow_rpm_controller_method(self):
        """Test the Crew.set_flow_rpm_controller method."""

        # Create test crew
        agent = Agent(role="Agent", goal="Goal", backstory="Backstory")
        task = Task(description="Task", agent=agent, expected_output="Output")
        crew = Crew(agents=[agent], tasks=[task], max_rpm=20)

        # Create RPM controller
        new_controller = RPMController(max_rpm=7)

        # Store original controllers
        original_crew_controller = crew._rpm_controller
        original_agent_controller = agent._rpm_controller

        # Set new controller
        crew.set_flow_rpm_controller(new_controller)

        # Verify changes
        assert crew._rpm_controller is new_controller
        assert crew._rpm_controller is not original_crew_controller
        assert agent._rpm_controller is new_controller
        assert agent._rpm_controller is not original_agent_controller

    def test_flow_cleanup_on_deletion(self):
        """Test that Flow properly cleans up RPM controller on deletion."""

        class TestFlow(Flow):
            @start()
            def test_method(self):
                return "test"

        flow = TestFlow(max_rpm=10)
        controller = flow._rpm_controller

        # Mock the stop_rpm_counter method to verify it's called
        with patch.object(controller, 'stop_rpm_counter') as mock_stop:
            # Delete the flow
            del flow

            # Verify cleanup was attempted
            mock_stop.assert_called_once()

    @pytest.mark.integration
    def test_integration_flow_with_rpm_control(self):
        """Integration test for complete flow with RPM control."""

        class IntegrationFlow(Flow):
            def __init__(self):
                super().__init__(max_rpm=3, verbose=True)

            @start()
            def create_and_run_crew(self):
                agent = Agent(
                    role="Integration Agent",
                    goal="Test integration",
                    backstory="Agent for integration testing"
                )

                task = Task(
                    description="Integration test task",
                    agent=agent,
                    expected_output="Integration test result"
                )

                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    max_rpm=50  # Should be overridden
                )

                return crew

        flow = IntegrationFlow()

        # Verify flow setup
        assert flow.max_rpm == 3
        assert flow._rpm_controller is not None
        assert flow._rpm_controller.max_rpm == 3

        # The actual crew execution would be tested in a full integration test
        # For unit testing, we verify the setup is correct

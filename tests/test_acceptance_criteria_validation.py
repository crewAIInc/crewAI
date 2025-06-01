"""Unit tests for acceptance criteria validation feature at task level."""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Tuple

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.agent_state import AgentState
from crewai.tools.agent_tools.scratchpad_tool import ScratchpadTool
from crewai.agents.parser import AgentFinish
from crewai.utilities import Printer
from crewai.llm import LLM


class TestAcceptanceCriteriaValidation:
    """Test suite for task-level acceptance criteria validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=LLM)
        self.mock_agent = MagicMock()
        self.mock_task = MagicMock()
        self.mock_crew = MagicMock()
        self.mock_tools_handler = MagicMock()

        # Set up agent attributes
        self.mock_agent.role = "Test Agent"
        self.mock_agent.reasoning = True
        self.mock_agent.verbose = False
        self.mock_agent.reasoning_interval = None
        self.mock_agent.adaptive_reasoning = False

        # Create executor
        self.executor = CrewAgentExecutor(
            llm=self.mock_llm,
            task=self.mock_task,
            crew=self.mock_crew,
            agent=self.mock_agent,
            prompt={},
            max_iter=10,
            tools=[],
            tools_names="",
            stop_words=[],
            tools_description="",
            tools_handler=self.mock_tools_handler,
            callbacks=[]
        )

        # Set up agent state with acceptance criteria
        self.executor.agent_state = AgentState(task_id="test-task-id")
        self.executor.agent_state.acceptance_criteria = [
            "Include all required information",
            "Format output properly",
            "Provide complete analysis"
        ]

        # Mock printer
        self.executor._printer = MagicMock(spec=Printer)

    def test_validate_acceptance_criteria_all_met(self):
        """Test validation when all acceptance criteria are met."""
        output = "Complete output with all information, properly formatted, with full analysis"

        # Configure LLM to return all criteria met
        self.mock_llm.call.return_value = '''{
            "1": "MET",
            "2": "MET",
            "3": "MET"
        }'''

        is_valid, unmet_criteria = self.executor._validate_acceptance_criteria(output)

        assert is_valid is True
        assert unmet_criteria == []
        assert self.mock_llm.call.call_count == 1

    def test_validate_acceptance_criteria_some_unmet(self):
        """Test validation when some criteria are not met."""
        output = "Partial output missing formatting"

        # Configure LLM to return mixed results
        self.mock_llm.call.return_value = '''{
            "1": "MET",
            "2": "NOT MET: Missing proper formatting",
            "3": "NOT MET: Analysis incomplete"
        }'''

        is_valid, unmet_criteria = self.executor._validate_acceptance_criteria(output)

        assert is_valid is False
        assert len(unmet_criteria) == 2
        assert "Format output properly" in unmet_criteria
        assert "Provide complete analysis" in unmet_criteria

    def test_create_criteria_retry_prompt_with_scratchpad(self):
        """Test retry prompt creation when scratchpad has data."""
        # Set up scratchpad tool with data
        self.executor.scratchpad_tool = ScratchpadTool()
        self.executor.agent_state.scratchpad = {
            "research_data": {"key": "value"},
            "analysis_results": ["item1", "item2"]
        }

        # Set up task details
        self.mock_task.description = "Analyze research data and provide insights"
        self.mock_task.expected_output = "A comprehensive report with analysis and recommendations"

        unmet_criteria = ["Include specific examples", "Add recommendations"]

        prompt = self.executor._create_criteria_retry_prompt(unmet_criteria)

        # Verify prompt content with new format
        assert "VALIDATION FAILED" in prompt
        assert "YOU CANNOT PROVIDE A FINAL ANSWER YET" in prompt
        assert "ORIGINAL TASK:" in prompt
        assert "Analyze research data" in prompt
        assert "EXPECTED OUTPUT:" in prompt
        assert "comprehensive report" in prompt
        assert "Include specific examples" in prompt
        assert "Add recommendations" in prompt
        assert "Access Scratchpad Memory" in prompt
        assert "'research_data'" in prompt
        assert "'analysis_results'" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt
        assert "CONTINUE WITH TOOL USAGE NOW" in prompt
        assert "DO NOT ATTEMPT ANOTHER FINAL ANSWER" in prompt

    def test_create_criteria_retry_prompt_without_scratchpad(self):
        """Test retry prompt creation when no scratchpad data exists."""
        unmet_criteria = ["Add more detail"]

        prompt = self.executor._create_criteria_retry_prompt(unmet_criteria)

        assert "Add more detail" in prompt
        assert "VALIDATION FAILED" in prompt
        assert "📦 YOUR SCRATCHPAD CONTAINS DATA" not in prompt

    @patch('crewai.agents.crew_agent_executor.get_llm_response')
    @patch('crewai.agents.crew_agent_executor.process_llm_response')
    def test_invoke_loop_blocks_incomplete_final_answer(self, mock_process, mock_get_response):
        """Test that invoke loop blocks incomplete final answers."""
        # Set up conditions
        self.executor.agent_state.acceptance_criteria = ["Complete all sections"]

        # First attempt returns incomplete final answer
        incomplete_answer = AgentFinish(
            thought="Done",
            output="Exploring potential follow-up tasks!",
            text="Final Answer: Exploring potential follow-up tasks!"
        )

        # After retry, return complete answer
        complete_answer = AgentFinish(
            thought="Done with all sections",
            output="Complete output with all sections addressed",
            text="Final Answer: Complete output with all sections addressed"
        )

        # Configure mocks
        mock_process.side_effect = [incomplete_answer, complete_answer]
        mock_get_response.return_value = "response"

        # Configure validation
        self.mock_llm.call.side_effect = [
            '{"1": "NOT MET: Missing required sections"}',  # First validation fails
            '{"1": "MET"}'  # Second validation passes
        ]

        # Execute
        result = self.executor._invoke_loop()

        # Verify
        assert result == complete_answer
        assert self.mock_llm.call.call_count == 2  # Two validation attempts
        assert mock_process.call_count == 2  # Two processing attempts

        # Verify error message was shown
        self._verify_validation_messages_shown()

    def test_validation_happens_on_every_final_answer_attempt(self):
        """Test that validation happens on every AgentFinish attempt."""
        self.executor.agent_state.acceptance_criteria = ["Complete all sections"]

        # Configure LLM to always return criteria not met
        self.mock_llm.call.return_value = '{"1": "NOT MET: Missing required sections"}'

        output = "Incomplete output"

        # Validate multiple times - each should trigger validation
        for _ in range(3):
            is_valid, unmet_criteria = self.executor._validate_acceptance_criteria(output)
            assert is_valid is False
            assert len(unmet_criteria) == 1

        # Verify validation was called every time
        assert self.mock_llm.call.call_count == 3

    def _verify_validation_messages_shown(self):
        """Helper to verify validation messages were displayed."""
        print_calls = self.executor._printer.print.call_args_list

        # Check for validation message
        validation_msg_shown = any(
            "Validating acceptance criteria" in str(call)
            for call in print_calls
        )

        # Check for failure message
        failure_msg_shown = any(
            "Cannot finalize" in str(call)
            for call in print_calls
        )

        assert validation_msg_shown or failure_msg_shown
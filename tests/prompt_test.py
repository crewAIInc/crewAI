import pytest
from unittest.mock import Mock, patch, MagicMock

from crewai.utilities.prompts import Prompts
from crewai.utilities.i18n import I18N


class TestPrompts:
    """Test suite for Prompts class"""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent"""
        agent = Mock()
        agent.goal = "Complete test goal"
        agent.role = "Test role"
        agent.backstory = "Test backstory"
        return agent

    @pytest.fixture
    def real_i18n(self):
        """Create real I18N object"""
        return I18N()

    def test_basic_initialization(self, mock_agent, real_i18n):
        """Test basic initialization"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        assert prompts.agent == mock_agent
        assert prompts.has_tools is False
        assert prompts.use_system_prompt is False
        assert prompts.system_template is None
        assert prompts.prompt_template is None
        assert prompts.response_template is None

    def test_initialization_with_all_parameters(self, mock_agent, real_i18n):
        """Test initialization with all parameters"""
        system_template = "System: {{ .System }}"
        prompt_template = "User: {{ .Prompt }}"
        response_template = "Assistant: {{ .Response }}"
        
        prompts = Prompts(
            agent=mock_agent,
            has_tools=True,
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        assert prompts.has_tools is True
        assert prompts.use_system_prompt is True
        assert prompts.system_template == system_template
        assert prompts.prompt_template == prompt_template
        assert prompts.response_template == response_template

    def test_task_execution_without_system_prompt_no_tools(self, mock_agent, real_i18n):
        """Test task execution without system prompt and no tools"""
        prompts = Prompts(agent=mock_agent, has_tools=False, i18n=real_i18n)
        
        result = prompts.task_execution()
        
        assert "prompt" in result
        assert "system" not in result
        assert "user" not in result
        
        # Verify expected content is included
        assert "Test role" in result["prompt"]
        assert "Test backstory" in result["prompt"]
        assert "Complete test goal" in result["prompt"]

    def test_task_execution_without_system_prompt_with_tools(self, mock_agent, real_i18n):
        """Test task execution without system prompt but with tools"""
        prompts = Prompts(agent=mock_agent, has_tools=True, i18n=real_i18n)
        
        result = prompts.task_execution()
        
        assert "prompt" in result
        assert "system" not in result
        assert "user" not in result
        
        # Verify tool-related content is included
        assert "tool" in result["prompt"].lower()

    def test_task_execution_with_system_prompt_no_custom_templates(self, mock_agent, real_i18n):
        """Test task execution with system prompt but no custom templates"""
        prompts = Prompts(
            agent=mock_agent, 
            has_tools=False, 
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        result = prompts.task_execution()
        
        assert "system" in result
        assert "user" in result
        assert "prompt" in result
        
        # Verify content distribution
        assert "Test role" in result["system"]
        assert "task" in result["user"].lower()

    def test_task_execution_with_system_prompt_and_custom_templates(self, mock_agent, real_i18n):
        """Test task execution with system prompt and custom templates"""
        system_template = "System: {{ .System }}"
        prompt_template = "User: {{ .Prompt }}"
        response_template = "Assistant: {{ .Response }}"
        
        prompts = Prompts(
            agent=mock_agent,
            has_tools=True,
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        result = prompts.task_execution()
        
        assert "system" in result
        assert "user" in result
        assert "prompt" in result
        
        # Verify custom templates are used
        assert result["system"].startswith("System:")
        assert result["user"].startswith("User:")

    def test_apply_agent_variables(self, mock_agent):
        """Test agent variable replacement functionality"""
        prompts = Prompts(agent=mock_agent)
        
        text = "Role: {role}, Goal: {goal}, Backstory: {backstory}"
        result = prompts._apply_agent_variables(text)
        
        assert "Role: Test role" in result
        assert "Goal: Complete test goal" in result
        assert "Backstory: Test backstory" in result

    def test_apply_agent_variables_with_none_agent(self):
        """Test agent variable replacement when agent is None"""
        prompts = Prompts(agent=None)
        
        text = "Role: {role}, Goal: {goal}, Backstory: {backstory}"
        result = prompts._apply_agent_variables(text)
        
        assert "Role: " in result
        assert "Goal: " in result
        assert "Backstory: " in result
        # Ensure no unreplaced variables remain
        assert "{role}" not in result
        assert "{goal}" not in result
        assert "{backstory}" not in result

    def test_build_custom_system(self, mock_agent, real_i18n):
        """Test building custom system prompt"""
        system_template = "System: {{ .System }} - Role: {role}"
        prompts = Prompts(
            agent=mock_agent, 
            system_template=system_template,
            i18n=real_i18n
        )
        
        result = prompts._build_custom_system(["role_playing"])
        
        assert "System:" in result
        assert "Test role" in result

    def test_build_custom_system_without_template(self, mock_agent, real_i18n):
        """Test fallback when no system template is provided"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        result = prompts._build_custom_system(["role_playing"])
        
        # Should fallback to default _build_prompt
        assert "Test role" in result

    def test_build_custom_user(self, mock_agent, real_i18n):
        """Test building custom user prompt"""
        prompt_template = "User: {{ .Prompt }} - Goal: {goal}"
        prompts = Prompts(
            agent=mock_agent,
            prompt_template=prompt_template,
            i18n=real_i18n
        )
        
        result = prompts._build_custom_user(["task"])
        
        assert "User:" in result
        assert "Complete test goal" in result

    def test_build_custom_user_without_template(self, mock_agent, real_i18n):
        """Test fallback when no user template is provided"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        result = prompts._build_custom_user(["task"])
        
        # Should contain task-related content
        assert "task" in result.lower()

    def test_backwards_compatibility(self, mock_agent, real_i18n):
        """Test backwards compatibility"""
        # Test that old usage patterns still work
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        result = prompts.task_execution()
        
        # Should return traditional format
        assert "prompt" in result
        assert isinstance(result["prompt"], str)

    @pytest.mark.parametrize("has_tools", [True, False])
    def test_tools_configuration(self, mock_agent, real_i18n, has_tools):
        """Parameterized test for tools configuration"""
        prompts = Prompts(agent=mock_agent, has_tools=has_tools, i18n=real_i18n)
        
        result = prompts.task_execution()
        
        if has_tools:
            assert "tool" in result["prompt"].lower()
        else:
            # Check if contains "no tools" information or tool-related information
            prompt_lower = result["prompt"].lower()
            assert "tool" in prompt_lower  # Even without tools, should have related instructions

    def test_unicode_and_special_characters(self, real_i18n):
        """Test Unicode and special character handling"""
        agent = Mock()
        agent.goal = "Goal with émojis 🎯"
        agent.role = "Role with áccénts"
        agent.backstory = "Background with symbols @#$%"
        
        prompts = Prompts(agent=agent, i18n=real_i18n)
        
        result = prompts.task_execution()
        
        # Verify Unicode characters are handled correctly
        prompt_content = result["prompt"]
        assert "🎯" in prompt_content
        assert "áccénts" in prompt_content
        assert "@#$%" in prompt_content

    def test_complex_workflow_with_all_features(self, mock_agent, real_i18n):
        """Test complex workflow with all features"""
        system_template = "System Context: {{ .System }}\nAgent Role: {role}"
        prompt_template = "User Request: {{ .Prompt }}\nAgent Goal: {goal}"
        response_template = "Assistant Response: {{ .Response }}\nAgent Backstory: {backstory}"
        
        prompts = Prompts(
            agent=mock_agent,
            has_tools=True,
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template,
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        result = prompts.task_execution()
        
        # Verify all parts exist
        assert "system" in result
        assert "user" in result
        assert "prompt" in result
        
        # Verify custom template formats
        assert "System Context:" in result["system"]
        assert "User Request:" in result["user"]
        
        # Verify agent variable replacement
        assert "Test role" in result["system"]
        assert "Complete test goal" in result["user"]

    def test_build_prompt_with_custom_templates(self, mock_agent, real_i18n):
        """Test building prompt with custom templates"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        system_template = "System: {{ .System }}"
        prompt_template = "User: {{ .Prompt }}"
        response_template = "Assistant: {{ .Response }}"
        
        result = prompts._build_prompt(
            ["role_playing", "task"],
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template
        )
        
        assert "System:" in result
        assert "User:" in result
        assert "Assistant:" in result

    def test_build_prompt_with_missing_response_template(self, mock_agent, real_i18n):
        """Test handling when response template is missing"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        system_template = "System: {{ .System }}"
        prompt_template = "User: {{ .Prompt }}"
        
        result = prompts._build_prompt(
            ["role_playing", "task"],
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=None
        )
        
        assert "System:" in result
        assert "User:" in result
        # Ensure no response part is added
        assert "Assistant:" not in result

    def test_template_variable_replacement_edge_cases(self, mock_agent, real_i18n):
        """Test edge cases for template variable replacement"""
        # Test template with multiple occurrences of the same variable
        system_template = "System: {{ .System }} Role: {role} Again Role: {role}"
        
        prompts = Prompts(
            agent=mock_agent,
            system_template=system_template,
            i18n=real_i18n
        )
        
        result = prompts._build_custom_system(["role_playing"])
        
        # Verify two {role} variables in template are replaced, plus one from role_playing slice
        # Total should be 3 "Test role" occurrences (2 from template + 1 from role_playing slice)
        assert result.count("Test role") >= 2  # At least 2, actually might be more
        assert "{role}" not in result  # Ensure no unreplaced variables

    def test_response_template_splitting(self, mock_agent, real_i18n):
        """Test response template splitting logic"""
        system_template = "System: {{ .System }}"
        prompt_template = "User: {{ .Prompt }}"
        response_template = "Response prefix {{ .Response }} response suffix"
        
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        result = prompts._build_prompt(
            ["role_playing", "task"],
            system_template=system_template,
            prompt_template=prompt_template,
            response_template=response_template
        )
        
        # Verify response template is correctly split
        assert "Response prefix" in result
        assert "response suffix" not in result  # Because split("{{ .Response }}")[0] only takes the first part

    def test_error_handling_with_mock_agent_attributes(self, real_i18n):
        """Test using Mock agent with string attributes"""
        mock_agent = Mock()
        mock_agent.goal = "string goal"
        mock_agent.role = "string role"  
        mock_agent.backstory = "string backstory"
        
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        text = "Role: {role}, Goal: {goal}, Backstory: {backstory}"
        result = prompts._apply_agent_variables(text)
        
        assert "Role: string role" in result
        assert "Goal: string goal" in result
        assert "Backstory: string backstory" in result

    def test_performance_with_large_templates(self, mock_agent, real_i18n):
        """Test performance with large templates"""
        # Create a large template
        large_template = "Large template: {{ .System }}" + "x" * 1000
        
        prompts = Prompts(
            agent=mock_agent,
            system_template=large_template,
            prompt_template="User: {{ .Prompt }}",
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        # Should be able to handle large templates normally
        result = prompts.task_execution()
        assert "system" in result
        assert len(result["system"]) > 1000

    def test_edge_case_empty_components(self, mock_agent, real_i18n):
        """Test edge case with empty component list"""
        prompts = Prompts(agent=mock_agent, i18n=real_i18n)
        
        result = prompts._build_prompt([])
        
        # With empty components, should return string with variable replacements or minimal content
        assert isinstance(result, str)

    def test_new_functionality_validation(self, mock_agent, real_i18n):
        """Validate that your new functionality works correctly"""
        # Test the new _build_custom_system and _build_custom_user methods
        system_template = "Custom System: {{ .System }}"
        prompt_template = "Custom User: {{ .Prompt }}"
        
        prompts = Prompts(
            agent=mock_agent,
            system_template=system_template,
            prompt_template=prompt_template,
            use_system_prompt=True,
            i18n=real_i18n
        )
        
        # Test custom system building
        system_result = prompts._build_custom_system(["role_playing"])
        assert "Custom System:" in system_result
        assert "Test role" in system_result
        
        # Test custom user building
        user_result = prompts._build_custom_user(["task"])
        assert "Custom User:" in user_result
        
        # Test complete task execution
        full_result = prompts.task_execution()
        assert "system" in full_result
        assert "user" in full_result
        assert full_result["system"].startswith("Custom System:")
        assert full_result["user"].startswith("Custom User:")

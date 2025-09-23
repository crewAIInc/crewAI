import pytest
from unittest.mock import Mock, patch
from crewai.llm import LLM
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task


class TestPromptCaching:
    """Test prompt caching functionality."""

    def test_llm_prompt_caching_disabled_by_default(self):
        """Test that prompt caching is disabled by default."""
        llm = LLM(model="gpt-4o")
        assert llm.enable_prompt_caching is False
        assert llm.cache_control == {"type": "ephemeral"}

    def test_llm_prompt_caching_enabled(self):
        """Test that prompt caching can be enabled."""
        llm = LLM(model="gpt-4o", enable_prompt_caching=True)
        assert llm.enable_prompt_caching is True

    def test_llm_custom_cache_control(self):
        """Test custom cache_control configuration."""
        custom_cache_control = {"type": "ephemeral", "ttl": 3600}
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True,
            cache_control=custom_cache_control
        )
        assert llm.cache_control == custom_cache_control

    def test_supports_prompt_caching_openai(self):
        """Test prompt caching support detection for OpenAI models."""
        llm = LLM(model="gpt-4o")
        assert llm._supports_prompt_caching() is True

    def test_supports_prompt_caching_anthropic(self):
        """Test prompt caching support detection for Anthropic models."""
        llm = LLM(model="anthropic/claude-3-5-sonnet-20240620")
        assert llm._supports_prompt_caching() is True

    def test_supports_prompt_caching_bedrock(self):
        """Test prompt caching support detection for Bedrock models."""
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        assert llm._supports_prompt_caching() is True

    def test_supports_prompt_caching_deepseek(self):
        """Test prompt caching support detection for Deepseek models."""
        llm = LLM(model="deepseek/deepseek-chat")
        assert llm._supports_prompt_caching() is True

    def test_supports_prompt_caching_unsupported(self):
        """Test prompt caching support detection for unsupported models."""
        llm = LLM(model="ollama/llama2")
        assert llm._supports_prompt_caching() is False

    def test_anthropic_cache_control_formatting_string_content(self):
        """Test that cache_control is properly formatted for Anthropic models with string content."""
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        system_message = next(m for m in formatted_messages if m["role"] == "system")
        assert isinstance(system_message["content"], list)
        assert system_message["content"][0]["type"] == "text"
        assert system_message["content"][0]["text"] == "You are a helpful assistant."
        assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}
        
        user_messages = [m for m in formatted_messages if m["role"] == "user"]
        actual_user_message = user_messages[1]  # Second user message is the actual one
        assert actual_user_message["content"] == "Hello, how are you?"

    def test_anthropic_cache_control_formatting_list_content(self):
        """Test that cache_control is properly formatted for Anthropic models with list content."""
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True
        )
        
        messages = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "text", "text": "Be concise and accurate."}
                ]
            },
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        system_message = next(m for m in formatted_messages if m["role"] == "system")
        assert isinstance(system_message["content"], list)
        assert len(system_message["content"]) == 2
        assert "cache_control" not in system_message["content"][0]
        assert system_message["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_multiple_system_messages_cache_control(self):
        """Test that cache_control is only added to the last system message."""
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True
        )
        
        messages = [
            {"role": "system", "content": "First system message."},
            {"role": "system", "content": "Second system message."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        first_system = formatted_messages[1]  # Index 1 after placeholder user message
        assert first_system["role"] == "system"
        assert first_system["content"] == "First system message."
        
        second_system = formatted_messages[2]  # Index 2 after placeholder user message
        assert second_system["role"] == "system"
        assert isinstance(second_system["content"], list)
        assert second_system["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_openai_prompt_caching_passthrough(self):
        """Test that OpenAI prompt caching works without message modification."""
        llm = LLM(model="gpt-4o", enable_prompt_caching=True)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        assert formatted_messages == messages

    def test_prompt_caching_disabled_passthrough(self):
        """Test that when prompt caching is disabled, messages pass through with normal Anthropic formatting."""
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=False
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        expected_messages = [
            {"role": "user", "content": "."},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        assert formatted_messages == expected_messages

    def test_unsupported_model_passthrough(self):
        """Test that unsupported models pass through messages unchanged even with caching enabled."""
        llm = LLM(
            model="ollama/llama2",
            enable_prompt_caching=True
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        formatted_messages = llm._format_messages_for_provider(messages)
        
        assert formatted_messages == messages

    @patch('crewai.llm.litellm.completion')
    def test_anthropic_cache_control_in_completion_call(self, mock_completion):
        """Test that cache_control is properly passed to litellm.completion for Anthropic models."""
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))],
            usage=Mock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        )
        
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        llm.call(messages)
        
        call_args = mock_completion.call_args[1]
        formatted_messages = call_args["messages"]
        
        system_message = next(m for m in formatted_messages if m["role"] == "system")
        assert isinstance(system_message["content"], list)
        assert system_message["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_crew_with_prompt_caching(self):
        """Test that crews can use LLMs with prompt caching enabled."""
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True
        )
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            llm=llm
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task])
        
        assert crew.agents[0].llm.enable_prompt_caching is True

    def test_bedrock_model_detection(self):
        """Test that Bedrock models are properly detected for prompt caching."""
        llm = LLM(
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            enable_prompt_caching=True
        )
        
        assert llm._supports_prompt_caching() is True
        assert llm.is_anthropic is False

    def test_custom_cache_control_parameters(self):
        """Test that custom cache_control parameters are properly stored."""
        custom_cache_control = {
            "type": "ephemeral",
            "max_age": 3600,
            "scope": "session"
        }
        
        llm = LLM(
            model="anthropic/claude-3-5-sonnet-20240620",
            enable_prompt_caching=True,
            cache_control=custom_cache_control
        )
        
        assert llm.cache_control == custom_cache_control
        
        messages = [{"role": "system", "content": "Test system message."}]
        formatted_messages = llm._format_messages_for_provider(messages)
        
        system_message = formatted_messages[1]
        assert isinstance(system_message["content"], list)
        assert system_message["content"][0]["cache_control"] == custom_cache_control

import pytest
from unittest.mock import Mock, patch
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.llm import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.tools_handler import ToolsHandler


class TestBedrockStopSequenceFix:
    """Test cases for issue #3317 - Bedrock stop sequence fix"""

    def test_stop_sequences_set_when_supported(self):
        """Test that stop sequences are set when LLM supports them"""
        mock_llm = Mock(spec=LLM)
        mock_llm.supports_stop_words.return_value = True
        mock_llm.stop = []
        
        mock_agent = Mock(spec=BaseAgent)
        mock_task = Mock()
        mock_crew = Mock()
        mock_tools_handler = Mock(spec=ToolsHandler)
        
        stop_words = ["\nObservation:", "\nThought:"]
        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=stop_words,
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor.use_stop_words is True
        assert mock_llm.stop == stop_words

    def test_stop_sequences_not_set_when_unsupported(self):
        """Test that stop sequences are not set when LLM doesn't support them"""
        mock_llm = Mock(spec=LLM)
        mock_llm.supports_stop_words.return_value = False
        mock_llm.stop = []
        
        mock_agent = Mock(spec=BaseAgent)
        mock_task = Mock()
        mock_crew = Mock()
        mock_tools_handler = Mock(spec=ToolsHandler)
        
        stop_words = ["\nObservation:", "\nThought:"]
        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=stop_words,
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor.use_stop_words is False
        assert mock_llm.stop == []

    def test_existing_stop_sequences_preserved_when_supported(self):
        """Test that existing stop sequences are preserved when adding new ones"""
        mock_llm = Mock(spec=LLM)
        mock_llm.supports_stop_words.return_value = True
        mock_llm.stop = ["existing_stop"]
        
        mock_agent = Mock(spec=BaseAgent)
        mock_task = Mock()
        mock_crew = Mock()
        mock_tools_handler = Mock(spec=ToolsHandler)
        
        stop_words = ["\nObservation:"]
        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=stop_words,
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor.use_stop_words is True
        assert "existing_stop" in mock_llm.stop
        assert "\nObservation:" in mock_llm.stop

    def test_existing_stop_sequences_preserved_when_unsupported(self):
        """Test that existing stop sequences are preserved when LLM doesn't support new ones"""
        mock_llm = Mock(spec=LLM)
        mock_llm.supports_stop_words.return_value = False
        mock_llm.stop = ["existing_stop"]
        
        mock_agent = Mock(spec=BaseAgent)
        mock_task = Mock()
        mock_crew = Mock()
        mock_tools_handler = Mock(spec=ToolsHandler)
        
        stop_words = ["\nObservation:"]
        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=stop_words,
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor.use_stop_words is False
        assert mock_llm.stop == ["existing_stop"]

    @patch('crewai.llm.get_supported_openai_params')
    def test_bedrock_model_stop_words_support(self, mock_get_params):
        """Test that Bedrock models correctly report stop word support"""
        mock_get_params.return_value = ['model', 'messages', 'temperature']  # No 'stop'
        
        llm = LLM(model="bedrock/converse/openai.gpt-oss-20b-1:0")
        
        assert llm.supports_stop_words() is False

    @patch('crewai.llm.get_supported_openai_params')
    def test_openai_model_stop_words_support(self, mock_get_params):
        """Test that OpenAI models correctly report stop word support"""
        mock_get_params.return_value = ['model', 'messages', 'temperature', 'stop']
        
        llm = LLM(model="gpt-4")
        
        assert llm.supports_stop_words() is True

    def test_use_stop_words_flag_consistency(self):
        """Test that use_stop_words flag is consistent with LLM support"""
        mock_llm_supporting = Mock(spec=LLM)
        mock_llm_supporting.supports_stop_words.return_value = True
        mock_llm_supporting.stop = []
        
        mock_agent = Mock(spec=BaseAgent)
        mock_task = Mock()
        mock_crew = Mock()
        mock_tools_handler = Mock(spec=ToolsHandler)
        
        executor_supporting = CrewAgentExecutor(
            llm=mock_llm_supporting,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=["\nObservation:"],
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor_supporting.use_stop_words is True
        
        mock_llm_non_supporting = Mock(spec=LLM)
        mock_llm_non_supporting.supports_stop_words.return_value = False
        mock_llm_non_supporting.stop = []
        
        executor_non_supporting = CrewAgentExecutor(
            llm=mock_llm_non_supporting,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt={"prompt": "test"},
            max_iter=5,
            tools=[],
            tools_names="",
            stop_words=["\nObservation:"],
            tools_description="",
            tools_handler=mock_tools_handler
        )
        
        assert executor_non_supporting.use_stop_words is False

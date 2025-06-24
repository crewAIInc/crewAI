from unittest.mock import Mock, patch
from crewai import Agent, LLM
from crewai.tasks.task_output import TaskOutput
from crewai.lite_agent import LiteAgentOutput
from crewai.utilities.xml_parser import (
    extract_xml_content,
    extract_all_xml_content,
    extract_multiple_xml_tags,
    extract_multiple_xml_tags_all,
    extract_xml_with_attributes,
    remove_xml_tags,
    strip_xml_tags_keep_content,
)


class TestLLMGenerationsLogprobs:
    """Test suite for LLM generations and logprobs functionality."""

    def test_llm_with_n_parameter(self):
        """Test that LLM accepts n parameter for multiple generations."""
        llm = LLM(model="gpt-3.5-turbo", n=3)
        assert llm.n == 3

    def test_llm_with_logprobs_parameter(self):
        """Test that LLM accepts logprobs parameter."""
        llm = LLM(model="gpt-3.5-turbo", logprobs=5)
        assert llm.logprobs == 5

    def test_llm_with_return_full_completion(self):
        """Test that LLM accepts return_full_completion parameter."""
        llm = LLM(model="gpt-3.5-turbo", return_full_completion=True)
        assert llm.return_full_completion is True

    def test_agent_with_llm_parameters(self):
        """Test that Agent accepts LLM generation parameters."""
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm_n=3,
            llm_logprobs=5,
            llm_top_logprobs=3,
            return_completion_metadata=True,
        )
        assert agent.llm_n == 3
        assert agent.llm_logprobs == 5
        assert agent.llm_top_logprobs == 3
        assert agent.return_completion_metadata is True

    @patch('crewai.llm.litellm.completion')
    def test_llm_call_returns_full_completion(self, mock_completion):
        """Test that LLM.call can return full completion object."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_response.model = "gpt-3.5-turbo"
        mock_response.created = 1234567890
        mock_response.id = "test-id"
        mock_response.object = "chat.completion"
        mock_response.system_fingerprint = "test-fingerprint"
        mock_completion.return_value = mock_response

        llm = LLM(model="gpt-3.5-turbo", return_full_completion=True)
        result = llm.call("Test message")

        assert isinstance(result, dict)
        assert result["content"] == "Test response"
        assert "choices" in result
        assert "usage" in result
        assert result["model"] == "gpt-3.5-turbo"

    def test_task_output_completion_metadata(self):
        """Test TaskOutput with completion metadata."""
        mock_choices = [
            Mock(message=Mock(content="Generation 1")),
            Mock(message=Mock(content="Generation 2")),
        ]
        mock_usage = {"prompt_tokens": 10, "completion_tokens": 15}
        
        completion_metadata = {
            "choices": mock_choices,
            "usage": mock_usage,
            "model": "gpt-3.5-turbo",
        }

        task_output = TaskOutput(
            description="Test task",
            raw="Generation 1",
            agent="test-agent",
            completion_metadata=completion_metadata,
        )

        generations = task_output.get_generations()
        assert generations == ["Generation 1", "Generation 2"]

        usage = task_output.get_usage_metrics()
        assert usage == mock_usage

    def test_lite_agent_output_completion_metadata(self):
        """Test LiteAgentOutput with completion metadata."""
        mock_choices = [
            Mock(message=Mock(content="Generation 1")),
            Mock(message=Mock(content="Generation 2")),
        ]
        mock_usage = {"prompt_tokens": 10, "completion_tokens": 15}
        
        completion_metadata = {
            "choices": mock_choices,
            "usage": mock_usage,
            "model": "gpt-3.5-turbo",
        }

        output = LiteAgentOutput(
            raw="Generation 1",
            agent_role="test-agent",
            completion_metadata=completion_metadata,
        )

        generations = output.get_generations()
        assert generations == ["Generation 1", "Generation 2"]

        usage = output.get_usage_metrics_from_completion()
        assert usage == mock_usage


class TestXMLParser:
    """Test suite for XML parsing functionality."""

    def test_extract_xml_content_basic(self):
        """Test basic XML content extraction."""
        text = "Some text <thinking>This is my thought</thinking> more text"
        result = extract_xml_content(text, "thinking")
        assert result == "This is my thought"

    def test_extract_xml_content_not_found(self):
        """Test XML content extraction when tag not found."""
        text = "Some text without the tag"
        result = extract_xml_content(text, "thinking")
        assert result is None

    def test_extract_xml_content_multiline(self):
        """Test XML content extraction with multiline content."""
        text = """Some text
        <thinking>
        This is a multiline
        thought process
        </thinking>
        more text"""
        result = extract_xml_content(text, "thinking")
        assert "multiline" in result
        assert "thought process" in result

    def test_extract_all_xml_content(self):
        """Test extracting all occurrences of XML content."""
        text = """
        <thinking>First thought</thinking>
        Some text
        <thinking>Second thought</thinking>
        """
        result = extract_all_xml_content(text, "thinking")
        assert len(result) == 2
        assert result[0] == "First thought"
        assert result[1] == "Second thought"

    def test_extract_multiple_xml_tags(self):
        """Test extracting multiple different XML tags."""
        text = """
        <thinking>My thoughts</thinking>
        <reasoning>My reasoning</reasoning>
        <conclusion>My conclusion</conclusion>
        """
        result = extract_multiple_xml_tags(text, ["thinking", "reasoning", "conclusion"])
        assert result["thinking"] == "My thoughts"
        assert result["reasoning"] == "My reasoning"
        assert result["conclusion"] == "My conclusion"

    def test_extract_multiple_xml_tags_all(self):
        """Test extracting all occurrences of multiple XML tags."""
        text = """
        <thinking>First thought</thinking>
        <reasoning>First reasoning</reasoning>
        <thinking>Second thought</thinking>
        """
        result = extract_multiple_xml_tags_all(text, ["thinking", "reasoning"])
        assert len(result["thinking"]) == 2
        assert len(result["reasoning"]) == 1
        assert result["thinking"][0] == "First thought"
        assert result["thinking"][1] == "Second thought"

    def test_extract_xml_with_attributes(self):
        """Test extracting XML with attributes."""
        text = '<thinking type="deep" level="2">Complex thought</thinking>'
        result = extract_xml_with_attributes(text, "thinking")
        assert len(result) == 1
        assert result[0]["content"] == "Complex thought"
        assert result[0]["attributes"]["type"] == "deep"
        assert result[0]["attributes"]["level"] == "2"

    def test_remove_xml_tags(self):
        """Test removing XML tags and their content."""
        text = "Keep this <thinking>Remove this</thinking> and this"
        result = remove_xml_tags(text, ["thinking"])
        assert result == "Keep this  and this"

    def test_strip_xml_tags_keep_content(self):
        """Test stripping XML tags but keeping content."""
        text = "Keep this <thinking>Keep this too</thinking> and this"
        result = strip_xml_tags_keep_content(text, ["thinking"])
        assert result == "Keep this Keep this too and this"

    def test_nested_xml_tags(self):
        """Test handling of nested XML tags."""
        text = "<outer>Before <inner>nested content</inner> after</outer>"
        result = extract_xml_content(text, "outer")
        assert "Before" in result
        assert "nested content" in result
        assert "after" in result

    def test_xml_with_special_characters(self):
        """Test XML parsing with special characters."""
        text = "<thinking>Content with & < > \" ' characters</thinking>"
        result = extract_xml_content(text, "thinking")
        assert "&" in result
        assert "<" in result
        assert ">" in result

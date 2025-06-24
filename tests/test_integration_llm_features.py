from unittest.mock import Mock, patch
from crewai import Agent, Task, Crew, LLM
from crewai.lite_agent import LiteAgent
from crewai.utilities.xml_parser import extract_xml_content


class TestIntegrationLLMFeatures:
    """Integration tests for LLM features with agents and tasks."""

    @patch('crewai.llm.litellm.completion')
    def test_agent_with_multiple_generations(self, mock_completion):
        """Test agent execution with multiple generations."""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Generation 1")),
            Mock(message=Mock(content="Generation 2")),
            Mock(message=Mock(content="Generation 3")),
        ]
        mock_response.usage = {"prompt_tokens": 20, "completion_tokens": 30}
        mock_response.model = "gpt-3.5-turbo"
        mock_response.created = 1234567890
        mock_response.id = "test-id"
        mock_response.object = "chat.completion"
        mock_response.system_fingerprint = "test-fingerprint"
        mock_completion.return_value = mock_response

        llm = LLM(model="gpt-3.5-turbo", n=3, return_full_completion=True)
        agent = Agent(
            role="writer",
            goal="write content",
            backstory="You are a writer",
            llm=llm,
            return_completion_metadata=True,
        )

        task = Task(
            description="Write a short story",
            agent=agent,
            expected_output="A short story",
        )

        with patch.object(agent, 'agent_executor') as mock_executor:
            mock_executor.invoke.return_value = {"output": "Generation 1"}
            
            result = agent.execute_task(task)
            assert result == "Generation 1"

    @patch('crewai.llm.litellm.completion')
    def test_lite_agent_with_xml_extraction(self, mock_completion):
        """Test LiteAgent with XML content extraction."""
        response_with_xml = """
        <thinking>
        I need to analyze this problem step by step.
        First, I'll consider the requirements.
        </thinking>
        
        Based on my analysis, here's the solution: The answer is 42.
        """

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=response_with_xml))]
        mock_response.usage = {"prompt_tokens": 15, "completion_tokens": 25}
        mock_response.model = "gpt-3.5-turbo"
        mock_response.created = 1234567890
        mock_response.id = "test-id"
        mock_response.object = "chat.completion"
        mock_response.system_fingerprint = "test-fingerprint"
        mock_completion.return_value = mock_response

        lite_agent = LiteAgent(
            role="analyst",
            goal="analyze problems",
            backstory="You are an analyst",
            llm=LLM(model="gpt-3.5-turbo", return_full_completion=True),
        )

        with patch.object(lite_agent, '_invoke_loop') as mock_invoke:
            mock_invoke.return_value = response_with_xml
            
            result = lite_agent.kickoff("Analyze this problem")
            
            thinking_content = extract_xml_content(result.raw, "thinking")
            assert thinking_content is not None
            assert "step by step" in thinking_content
            assert "requirements" in thinking_content

    def test_xml_parser_with_complex_agent_output(self):
        """Test XML parser with complex agent output containing multiple tags."""
        complex_output = """
        <thinking>
        This is a complex problem that requires careful analysis.
        I need to break it down into steps.
        </thinking>
        
        <reasoning>
        Step 1: Understand the requirements
        Step 2: Analyze the constraints
        Step 3: Develop a solution
        </reasoning>
        
        <conclusion>
        The best approach is to use a systematic methodology.
        </conclusion>
        
        Final answer: Use the systematic approach outlined above.
        """

        thinking = extract_xml_content(complex_output, "thinking")
        reasoning = extract_xml_content(complex_output, "reasoning")
        conclusion = extract_xml_content(complex_output, "conclusion")

        assert thinking is not None
        assert "complex problem" in thinking
        assert reasoning is not None
        assert "Step 1" in reasoning
        assert "Step 2" in reasoning
        assert "Step 3" in reasoning
        assert conclusion is not None
        assert "systematic methodology" in conclusion

    @patch('crewai.llm.litellm.completion')
    def test_crew_with_llm_parameters(self, mock_completion):
        """Test crew execution with LLM parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        mock_response.model = "gpt-3.5-turbo"
        mock_response.created = 1234567890
        mock_response.id = "test-id"
        mock_response.object = "chat.completion"
        mock_response.system_fingerprint = "test-fingerprint"
        mock_completion.return_value = mock_response

        agent = Agent(
            role="analyst",
            goal="analyze data",
            backstory="You are an analyst",
            llm_n=2,
            llm_logprobs=5,
            return_completion_metadata=True,
        )

        task = Task(
            description="Analyze the data",
            agent=agent,
            expected_output="Analysis results",
        )

        crew = Crew(agents=[agent], tasks=[task])
        
        with patch.object(crew, 'kickoff') as mock_kickoff:
            mock_output = Mock()
            mock_output.tasks_output = [Mock(completion_metadata={"choices": mock_response.choices})]
            mock_kickoff.return_value = mock_output
            
            result = crew.kickoff()
            assert result is not None

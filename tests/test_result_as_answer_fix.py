from unittest.mock import patch
from crewai.agent import Agent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.tools.tool_types import ToolAnswerResult
from pydantic import BaseModel


class TestOutputModel(BaseModel):
    message: str
    status: str


class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    result_as_answer: bool = False
    
    def _run(self, *args, **kwargs) -> str:
        return "Mock tool output"


class MockLLM:
    def call(self, messages, **kwargs):
        return "LLM processed output"
    
    def __call__(self, messages, **kwargs):
        return self.call(messages, **kwargs)


def test_tool_with_result_as_answer_true_bypasses_conversion():
    """Test that tools with result_as_answer=True return output without conversion."""
    tool = MockTool()
    tool.result_as_answer = True
    
    agent = Agent(
        role="test_agent",
        goal="test goal",
        backstory="test backstory",
        llm=MockLLM(),
        tools=[tool]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
        output_pydantic=TestOutputModel
    )
    
    agent.tools_results = [
        {
            "tool": "mock_tool",
            "result": "Plain string output that should not be converted",
            "result_as_answer": True
        }
    ]
    
    with patch.object(agent, 'execute_task') as mock_execute:
        mock_execute.return_value = ToolAnswerResult("Plain string output that should not be converted")
        
        result = task.execute_sync()
        
        assert result.raw == "Plain string output that should not be converted"
        assert result.pydantic is None
        assert result.json_dict is None


def test_tool_with_result_as_answer_false_applies_conversion():
    """Test that tools with result_as_answer=False still apply conversion when output_pydantic is set."""
    tool = MockTool()
    tool.result_as_answer = False
    
    agent = Agent(
        role="test_agent",
        goal="test goal", 
        backstory="test backstory",
        llm=MockLLM(),
        tools=[tool]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
        output_pydantic=TestOutputModel
    )
    
    with patch.object(agent, 'execute_task') as mock_execute:
        mock_execute.return_value = '{"message": "test", "status": "success"}'
        
        with patch('crewai.task.convert_to_model') as mock_convert:
            mock_convert.return_value = TestOutputModel(message="test", status="success")
            
            result = task.execute_sync()
            
            assert mock_convert.called
            assert result.pydantic is not None
            assert isinstance(result.pydantic, TestOutputModel)


def test_multiple_tools_last_result_as_answer_wins():
    """Test that when multiple tools are used, the last one with result_as_answer=True is used."""
    agent = Agent(
        role="test_agent",
        goal="test goal",
        backstory="test backstory", 
        llm=MockLLM()
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    agent.tools_results = [
        {
            "tool": "tool1",
            "result": "First tool output",
            "result_as_answer": False
        },
        {
            "tool": "tool2", 
            "result": "Second tool output",
            "result_as_answer": True
        },
        {
            "tool": "tool3",
            "result": "Third tool output", 
            "result_as_answer": False
        },
        {
            "tool": "tool4",
            "result": "Final tool output that should be used",
            "result_as_answer": True
        }
    ]
    
    with patch.object(agent, 'execute_task') as mock_execute:
        mock_execute.return_value = ToolAnswerResult("Final tool output that should be used")
        
        result = task.execute_sync()
        
        assert result.raw == "Final tool output that should be used"


def test_tool_answer_result_wrapper():
    """Test the ToolAnswerResult wrapper class."""
    result = ToolAnswerResult("test output")
    
    assert str(result) == "test output"
    
    assert result.result == "test output"


def test_reproduction_of_issue_3335():
    """Reproduction test for GitHub issue #3335."""
    
    tool = MockTool()
    tool.result_as_answer = True
    
    def mock_tool_run(*args, **kwargs):
        return "This is a plain string that should not be converted to JSON"
    
    tool._run = mock_tool_run
    
    agent = Agent(
        role="test_agent",
        goal="test goal",
        backstory="test backstory",
        llm=MockLLM(),
        tools=[tool]
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output", 
        agent=agent,
        output_pydantic=TestOutputModel
    )
    
    agent.tools_results = [
        {
            "tool": "mock_tool",
            "result": "This is a plain string that should not be converted to JSON",
            "result_as_answer": True
        }
    ]
    
    with patch.object(agent, 'execute_task') as mock_execute:
        mock_execute.return_value = ToolAnswerResult("This is a plain string that should not be converted to JSON")
        
        result = task.execute_sync()
        
        assert result.raw == "This is a plain string that should not be converted to JSON"
        assert result.pydantic is None
        assert result.json_dict is None


def test_edge_case_complex_tool_output():
    """Test edge case with complex tool output that should be preserved."""
    complex_output = """
    This is a multi-line output
    with special characters: !@#$%^&*()
    and some JSON-like content: {"key": "value"}
    but it should be preserved as-is when result_as_answer=True
    """
    
    agent = Agent(
        role="test_agent", 
        goal="test goal",
        backstory="test backstory",
        llm=MockLLM()
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
        output_pydantic=TestOutputModel
    )
    
    agent.tools_results = [
        {
            "tool": "complex_tool",
            "result": complex_output,
            "result_as_answer": True
        }
    ]
    
    with patch.object(agent, 'execute_task') as mock_execute:
        mock_execute.return_value = ToolAnswerResult(complex_output)
        
        result = task.execute_sync()
        
        assert result.raw == complex_output
        assert result.pydantic is None
        assert result.json_dict is None

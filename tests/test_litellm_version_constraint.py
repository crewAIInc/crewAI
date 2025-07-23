import pytest
import importlib.metadata
from packaging import version

from crewai.llm import LLM
from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew


def test_litellm_minimum_version_constraint():
    """Test that litellm meets the minimum version requirement."""
    try:
        litellm_version = importlib.metadata.version("litellm")
        minimum_version = "1.74.3"
        
        assert version.parse(litellm_version) >= version.parse(minimum_version), (
            f"litellm version {litellm_version} is below minimum required version {minimum_version}"
        )
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("litellm package is not installed")


def test_llm_creation_with_relaxed_litellm_constraint():
    """Test that LLM can be created successfully with the relaxed litellm constraint."""
    llm = LLM(model="gpt-4o-mini")
    assert llm is not None
    assert llm.model == "gpt-4o-mini"


def test_basic_llm_functionality_with_relaxed_constraint():
    """Test that basic LLM functionality works with the relaxed litellm constraint."""
    llm = LLM(model="gpt-4o-mini", temperature=0.7, max_tokens=100)
    
    assert llm.model == "gpt-4o-mini"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 100


def test_agent_creation_with_relaxed_litellm_constraint():
    """Test that Agent can be created with LLM using relaxed litellm constraint."""
    llm = LLM(model="gpt-4o-mini")
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm=llm
    )
    
    assert agent is not None
    assert agent.llm == llm
    assert agent.role == "Test Agent"


def test_crew_functionality_with_relaxed_litellm_constraint():
    """Test that Crew functionality works with the relaxed litellm constraint."""
    llm = LLM(model="gpt-4o-mini")
    agent = Agent(
        role="Test Agent",
        goal="Test goal", 
        backstory="Test backstory",
        llm=llm
    )
    
    task = Task(
        description="Test task description",
        expected_output="Test output",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    
    assert crew is not None
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    assert crew.agents[0] == agent
    assert crew.tasks[0] == task


def test_litellm_import_functionality():
    """Test that litellm can be imported and basic functionality works."""
    import litellm
    from litellm.exceptions import ContextWindowExceededError, AuthenticationError
    
    assert hasattr(litellm, 'completion')
    assert ContextWindowExceededError is not None
    assert AuthenticationError is not None


def test_llm_supports_function_calling():
    """Test that LLM function calling support detection works with relaxed constraint."""
    llm = LLM(model="gpt-4o-mini")
    
    supports_functions = llm.supports_function_calling()
    assert isinstance(supports_functions, bool)


def test_llm_context_window_size():
    """Test that LLM context window size detection works with relaxed constraint."""
    llm = LLM(model="gpt-4o-mini")
    
    context_window = llm.get_context_window_size()
    assert isinstance(context_window, int)
    assert context_window > 0


def test_llm_anthropic_model_detection():
    """Test that Anthropic model detection works with relaxed constraint."""
    anthropic_llm = LLM(model="anthropic/claude-3-sonnet")
    openai_llm = LLM(model="gpt-4o-mini")
    
    assert anthropic_llm._is_anthropic_model() is True
    assert openai_llm._is_anthropic_model() is False

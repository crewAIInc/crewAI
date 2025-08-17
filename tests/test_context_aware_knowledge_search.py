"""Test context-aware knowledge search functionality."""

import pytest
from unittest.mock import patch

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


@pytest.mark.vcr(filter_headers=["authorization"])
def test_knowledge_search_with_context():
    """Test that knowledge search includes context from previous tasks."""
    content = "The company's main product is a CRM system. The system has three modules: Sales, Marketing, and Support."
    string_source = StringKnowledgeSource(content=content)
    
    researcher = Agent(
        role="Research Analyst",
        goal="Research company information",
        backstory="You are a research analyst.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Write content based on research",
        backstory="You are a content writer.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    research_task = Task(
        description="Research the company's main product",
        expected_output="A summary of the company's main product",
        agent=researcher,
    )
    
    writing_task = Task(
        description="Write a detailed description of the CRM modules",
        expected_output="A detailed description of each CRM module",
        agent=writer,
        context=[research_task],
    )
    
    crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
    
    with patch.object(writer, '_get_knowledge_search_query') as mock_search:
        mock_search.return_value = "mocked query"
        crew.kickoff()
        
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert len(call_args[0]) == 2
        assert call_args[0][1] is not None
        assert "CRM system" in call_args[0][1] or "product" in call_args[0][1]


@pytest.mark.vcr(filter_headers=["authorization"])
def test_knowledge_search_without_context():
    """Test that knowledge search works without context (backward compatibility)."""
    content = "The company's main product is a CRM system."
    string_source = StringKnowledgeSource(content=content)
    
    agent = Agent(
        role="Research Analyst",
        goal="Research company information",
        backstory="You are a research analyst.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    task = Task(
        description="Research the company's main product",
        expected_output="A summary of the company's main product",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    with patch.object(agent, '_get_knowledge_search_query') as mock_search:
        mock_search.return_value = "mocked query"
        crew.kickoff()
        
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert len(call_args[0]) == 2
        assert call_args[0][1] == ""


@pytest.mark.vcr(filter_headers=["authorization"])
def test_context_aware_knowledge_search_integration():
    """Integration test for context-aware knowledge search."""
    knowledge_content = """
    Project Alpha is a web application built with React and Node.js.
    Project Beta is a mobile application built with React Native.
    The team uses Agile methodology with 2-week sprints.
    The database is PostgreSQL with Redis for caching.
    """
    
    string_source = StringKnowledgeSource(content=knowledge_content)
    
    project_manager = Agent(
        role="Project Manager",
        goal="Gather project information",
        backstory="You manage software projects.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    tech_lead = Agent(
        role="Technical Lead",
        goal="Provide technical details",
        backstory="You are a technical expert.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    project_overview_task = Task(
        description="Provide an overview of Project Alpha",
        expected_output="Overview of Project Alpha including its technology stack",
        agent=project_manager,
    )
    
    technical_details_task = Task(
        description="Provide technical implementation details for the project",
        expected_output="Technical implementation details including database and caching",
        agent=tech_lead,
        context=[project_overview_task],
    )
    
    crew = Crew(agents=[project_manager, tech_lead], tasks=[project_overview_task, technical_details_task])
    
    result = crew.kickoff()
    
    assert result.raw is not None
    assert any(keyword in result.raw.lower() for keyword in ["react", "node", "postgresql", "redis"])


def test_knowledge_search_query_template_with_context():
    """Test that the knowledge search query template includes context properly."""
    agent = Agent(
        role="Test Agent",
        goal="Test knowledge search",
        backstory="Test agent",
        llm=LLM(model="gpt-4o-mini"),
    )
    
    task_prompt = "What is the main product?"
    context = "Previous research shows the company focuses on CRM solutions."
    
    with patch.object(agent.llm, 'call') as mock_call:
        mock_call.return_value = "mocked response"
        
        agent._get_knowledge_search_query(task_prompt, context)
        
        mock_call.assert_called_once()
        call_args = mock_call.call_args[0][0]
        user_message = call_args[1]['content']
        
        assert task_prompt in user_message
        assert context in user_message
        assert "Context from previous tasks:" in user_message


def test_knowledge_search_query_template_without_context():
    """Test that the knowledge search query template works without context."""
    agent = Agent(
        role="Test Agent",
        goal="Test knowledge search",
        backstory="Test agent",
        llm=LLM(model="gpt-4o-mini"),
    )
    
    task_prompt = "What is the main product?"
    
    with patch.object(agent.llm, 'call') as mock_call:
        mock_call.return_value = "mocked response"
        
        agent._get_knowledge_search_query(task_prompt)
        
        mock_call.assert_called_once()
        call_args = mock_call.call_args[0][0]
        user_message = call_args[1]['content']
        
        assert task_prompt in user_message
        assert "Context from previous tasks:" not in user_message


def test_structured_context_integration():
    """Test context-aware knowledge search with structured context data."""
    knowledge_content = """
    Error URS-01: User registration service unavailable.
    Method getUserStatus returns user account status.
    API endpoint /api/users/{id}/status for user status queries.
    Database table user_accounts stores user information.
    """
    
    string_source = StringKnowledgeSource(content=knowledge_content)
    
    agent = Agent(
        role="Technical Support",
        goal="Resolve technical issues",
        backstory="You help resolve technical problems.",
        llm=LLM(model="gpt-4o-mini"),
        knowledge_sources=[string_source],
    )
    
    task_prompt = "How to resolve the user status error?"
    structured_context = '{"method": "getUserStatus", "error_code": "URS-01", "endpoint": "/api/users/{id}/status"}'
    
    with patch.object(agent.llm, 'call') as mock_call:
        mock_call.return_value = "Check getUserStatus method and URS-01 error"
        
        agent._get_knowledge_search_query(task_prompt, structured_context)
        
        mock_call.assert_called_once()
        call_args = mock_call.call_args[0][0]
        user_message = call_args[1]['content']
        
        assert task_prompt in user_message
        assert "getUserStatus" in user_message
        assert "URS-01" in user_message
        assert "Context from previous tasks:" in user_message

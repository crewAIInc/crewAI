import pytest
from unittest.mock import MagicMock, patch

from crewai import Agent, Crew, LLM, Process, Task
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


def test_agent_without_knowledge_uses_crew_knowledge():
    """Test that an agent without knowledge sources can use crew knowledge sources."""
    content = "John is 30 years old and lives in San Francisco."
    string_source = StringKnowledgeSource(content=content)

    agent = Agent(
        role="Information Agent",
        goal="Provide information based on knowledge sources",
        backstory="You have access to specific knowledge sources.",
        llm=LLM(model="gpt-4o-mini"),
    )

    task = Task(
        description="How old is John and where does he live?",
        expected_output="John's age and location.",
        agent=agent,
    )

    with patch('crewai.knowledge.knowledge.Knowledge.query', return_value=[{"context": content}]) as mock_query:
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            knowledge_sources=[string_source],
        )
        
        agent.crew = crew
        
        with patch.object(Agent, '_get_knowledge_search_query', return_value="test query"):
            with patch.object(Agent, '_execute_without_timeout', return_value="John is 30 years old and lives in San Francisco."):
                result = agent.execute_task(task)
                
                assert mock_query.called
                
                assert hasattr(agent, 'crew_knowledge_context')
                
                assert "John" in result

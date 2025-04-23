"""Integration test for Elasticsearch with CrewAI."""

import os
import unittest

import pytest

from crewai import Agent, Crew, Task
from crewai.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


@pytest.mark.skipif(
    os.environ.get("RUN_ELASTICSEARCH_TESTS") != "true",
    reason="Elasticsearch tests require RUN_ELASTICSEARCH_TESTS=true"
)
class TestElasticsearchIntegration(unittest.TestCase):
    """Integration test for Elasticsearch with CrewAI."""

    def test_crew_with_elasticsearch_memory(self):
        """Test a crew with Elasticsearch memory."""
        researcher = Agent(
            role="Researcher",
            goal="Research a topic",
            backstory="You are a researcher who loves to find information.",
        )
        
        writer = Agent(
            role="Writer",
            goal="Write about a topic",
            backstory="You are a writer who loves to write about topics.",
        )
        
        research_task = Task(
            description="Research about AI",
            expected_output="Information about AI",
            agent=researcher,
        )
        
        write_task = Task(
            description="Write about AI",
            expected_output="Article about AI",
            agent=writer,
            context=[research_task],
        )
        
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            memory_config={"provider": "elasticsearch"},
        )
        
        result = crew.kickoff()
        
        self.assertIsNotNone(result)
        
    def test_crew_with_elasticsearch_knowledge(self):
        """Test a crew with Elasticsearch knowledge."""
        content = "AI is a field of computer science that focuses on creating machines that can perform tasks that typically require human intelligence."
        string_source = StringKnowledgeSource(
            content=content, metadata={"topic": "AI"}
        )
        
        knowledge = Knowledge(
            collection_name="test",
            sources=[string_source],
            storage_provider="elasticsearch",
        )
        
        agent = Agent(
            role="AI Expert",
            goal="Explain AI",
            backstory="You are an AI expert who loves to explain AI concepts.",
            knowledge=[knowledge],
        )
        
        task = Task(
            description="Explain what AI is",
            expected_output="Explanation of AI",
            agent=agent,
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
        )
        
        result = crew.kickoff()
        
        self.assertIsNotNone(result)

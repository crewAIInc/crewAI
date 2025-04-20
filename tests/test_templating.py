import datetime
from typing import Dict, List

import pytest

from crewai.agent import Agent
from crewai.task import Task


class TestTemplating:
    def test_task_interpolation(self):
        task = Task(
            description="Research about {topic} and provide {count} key points",
            expected_output="A list of {count} key points about {topic}"
        )
        
        inputs = {"topic": "AI", "count": 5}
        task.interpolate_inputs(inputs)
        
        assert task.description == "Research about AI and provide 5 key points"
        assert task.expected_output == "A list of 5 key points about AI"
        
        task = Task(
            description="Research about {topics[0]} and {topics[1]}",
            expected_output="Analysis of {{data.main_theme}}"
        )
        
        inputs = {
            "topics": ["AI", "Machine Learning"],
            "data": {"main_theme": "Technology Trends"}
        }
        
        task.interpolate_inputs(inputs)
        
        assert task.description == "Research about AI and Machine Learning"
        assert task.expected_output == "Analysis of Technology Trends"
    
    def test_agent_interpolation(self):
        agent = Agent(
            role="{industry} Researcher",
            goal="Research {count} key developments in {industry}",
            backstory="You are a senior researcher in the {industry} field with {experience} years of experience"
        )
        
        inputs = {"industry": "Healthcare", "count": 5, "experience": 10}
        agent.interpolate_inputs(inputs)
        
        assert agent.role == "Healthcare Researcher"
        assert agent.goal == "Research 5 key developments in Healthcare"
        assert agent.backstory == "You are a senior researcher in the Healthcare field with 10 years of experience"
        
        agent = Agent(
            role="{{specialties[0]}} and {{specialties[1]}} Specialist",
            goal="Analyze trends in {{fields.primary}} sector",
            backstory="Expert in {{fields.primary}} and {{fields.secondary}}"
        )
        
        inputs = {
            "specialties": ["AI", "Data Science"],
            "fields": {"primary": "Healthcare", "secondary": "Finance"}
        }
        
        agent.interpolate_inputs(inputs)
        
        assert agent.role == "AI and Data Science Specialist"
        assert agent.goal == "Analyze trends in Healthcare sector"
        assert agent.backstory == "Expert in Healthcare and Finance"
    
    def test_conditional_templating(self):
        task = Task(
            description="{% if priority == 'high' %}URGENT: {% endif %}Research {topic}",
            expected_output="A report on {topic}"
        )
        
        inputs = {"topic": "AI", "priority": "high"}
        task.interpolate_inputs(inputs)
        assert task.description == "URGENT: Research AI"
        
        inputs = {"topic": "AI", "priority": "low"}
        task.interpolate_inputs(inputs)
        assert task.description == "Research AI"
    
    def test_loop_templating(self):
        task = Task(
            description="Research the following topics: {% for topic in topics %}{{topic}}{% if not loop.last %}, {% endif %}{% endfor %}",
            expected_output="A report on multiple topics"
        )
        
        inputs = {"topics": ["AI", "Machine Learning", "Data Science"]}
        task.interpolate_inputs(inputs)
        assert task.description == "Research the following topics: AI, Machine Learning, Data Science"

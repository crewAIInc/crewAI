import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task


def test_agent_default_language():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )
    assert agent.i18n.language == "en"


def test_agent_with_spanish_language():
    agent = Agent(
        role="Agente de Prueba",
        goal="Objetivo de prueba",
        backstory="Historia de fondo de prueba",
        language="es"
    )
    assert agent.i18n.language == "es"
    assert "Eres {role}" in agent.i18n.slice("role_playing")


def test_agent_with_english_language_explicit():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        language="en"
    )
    assert agent.i18n.language == "en"
    assert "You are {role}" in agent.i18n.slice("role_playing")


def test_crew_default_language():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task])
    assert crew.language is None


def test_crew_with_spanish_language():
    agent = Agent(
        role="Agente de Prueba",
        goal="Objetivo de prueba",
        backstory="Historia de fondo de prueba"
    )
    task = Task(
        description="Tarea de prueba",
        expected_output="Salida de prueba",
        agent=agent
    )
    crew = Crew(agents=[agent], tasks=[task], language="es")
    assert crew.language == "es"


def test_crew_language_propagates_to_agents():
    agent1 = Agent(
        role="Agent 1",
        goal="Goal 1",
        backstory="Backstory 1"
    )
    agent2 = Agent(
        role="Agent 2",
        goal="Goal 2",
        backstory="Backstory 2"
    )
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent1
    )
    crew = Crew(agents=[agent1, agent2], tasks=[task], language="es")
    
    assert crew.language == "es"


def test_agent_language_overrides_default():
    agent_en = Agent(
        role="English Agent",
        goal="English goal",
        backstory="English backstory",
        language="en"
    )
    agent_es = Agent(
        role="Spanish Agent",
        goal="Spanish goal",
        backstory="Spanish backstory",
        language="es"
    )
    
    assert agent_en.i18n.language == "en"
    assert agent_es.i18n.language == "es"
    assert "You are {role}" in agent_en.i18n.slice("role_playing")
    assert "Eres {role}" in agent_es.i18n.slice("role_playing")


def test_agent_without_language_uses_default():
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )
    assert agent.language is None
    assert agent.i18n.language == "en"

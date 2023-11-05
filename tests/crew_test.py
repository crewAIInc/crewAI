"""Test Agent creation and execution basic functionality."""

import pytest
from ..crewai import Agent, Crew, Task, Process

def test_crew_creation():
	agent_CTO = Agent(
		role="CTO",
		goal="Help your team craft the most amazing apps ever made.",
		backstory="You're world class CTO that works on the best web consulting agency."
	)
	agent_QA = Agent(
		role="QA Engineer",
		goal="Make sure ship the best software possible with the highest quality",
		backstory="You're the best at QA in the whole team, you are known for catching all bugs and advocate for improvements."
	)
	agent_Eng = Agent(
		role="Web Engineer",
		goal="Build amazing websites by writing high quality html, css and js.",
		backstory="You're great at vanila JS, CSS and HTMl, you got hired to build amazing website using your skills."
	)

	task = Task(
		description="Build a landing page for a website that sells dog food."
	)

	crew = Crew(
		agents=[agent_CTO, agent_Eng, agent_QA],
		goal="Build amazing landing pages.",
		tasks=[task],
		process=Process.consensual
	)

	assert crew.kickoff() == 'lol'
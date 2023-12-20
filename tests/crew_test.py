"""Test Agent creation and execution basic functionality."""

import json
import pytest
from ..crewai import Agent, Crew, Task, Process

ceo = Agent(
	role="CEO",
	goal="Make sure the writers in your company produce amazing content.",
	backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
	allow_delegation=True
)

researcher = Agent(
	role="Researcher",
	goal="Make the best research and analysis on content about AI and AI agents",
	backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
	allow_delegation=False
)

writer = Agent(
	role="Senior Writer",
	goal="Write the best content about AI and AI agents.",
	backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
	allow_delegation=False
)

def test_crew_config_conditional_requirement():
	with pytest.raises(ValueError):
		Crew(process=Process.sequential)
	
	config = json.dumps({
		"agents": [
			{
				"role": "Senior Researcher",
				"goal": "Make the best research and analysis on content about AI and AI agents",
				"backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer."
			},
			{
				"role": "Senior Writer",
				"goal": "Write the best content about AI and AI agents.",
				"backstory": "You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer."
			}
		],
		"tasks": [
			{
				"description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
				"agent": "Senior Researcher"
			},
			{
				"description": "Write a 1 amazing paragraph highlight for each idead that showcases how good an article about this topic could be, check references if necessary or search for more content but make sure it's unique, interesting and well written. Return the list of ideas with their paragraph and your notes.",
				"agent": "Senior Writer"
			}
		]		
	})
	parsed_config = json.loads(config)

	try:
			crew = Crew(process=Process.sequential, config=config)			
	except ValueError:
			pytest.fail("Unexpected ValidationError raised")
	
	assert [agent.role for agent in crew.agents] == [agent['role'] for agent in parsed_config['agents']]
	assert [task.description for task in crew.tasks] == [task['description'] for task in parsed_config['tasks']]

def test_crew_config_with_wrong_keys():
	no_tasks_config = json.dumps({
		"agents": [
			{
				"role": "Senior Researcher",
				"goal": "Make the best research and analysis on content about AI and AI agents",
				"backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer."
			}
		]
	})

	no_agents_config = json.dumps({
		"tasks": [
			{
				"description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
				"agent": "Senior Researcher"
			}
		]		
	})
	with pytest.raises(ValueError):
		Crew(process=Process.sequential, config='{"wrong_key": "wrong_value"}')
	with pytest.raises(ValueError):
		Crew(process=Process.sequential, config=no_tasks_config)
	with pytest.raises(ValueError):
		Crew(process=Process.sequential, config=no_agents_config)

@pytest.mark.vcr()
def test_crew_creation():
	tasks = [
		Task(
			description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
			agent=researcher
		),
		Task(
			description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
			agent=writer
		)
	]

	crew = Crew(
		agents=[researcher, writer],
		process=Process.sequential,
		tasks=tasks,
	)

	assert crew.kickoff() == """1. AI and Ethical Dilemmas: 
Facing the future, we grapple with moral quandaries brought forth by AI. This article will delve into the ethical dilemmas posed by AI, such as data privacy, algorithmic bias, and the responsibility of AI’s decisions. It will provide a compelling narrative about the necessity of considering ethical aspects in AI development and how they might shape the future of humanity.

2. AI in Healthcare:
Imagine a world where diagnosis and treatment are not limited by human errors or geographical boundaries. This article will explore the transformative role of AI in healthcare, from robotic surgeries to personalized medicine. It will offer fascinating insights into how AI can revolutionize healthcare, save countless lives, and bring about a new era of medical science.

3. The Role of AI in Climate Change:
As our planet faces an unprecedented climate crisis, we turn to AI for solutions. This article will illuminate the role of AI in combating climate change by optimizing renewable energy, predicting weather patterns, and managing resources. It will underscore the pivotal role that AI plays in our fight against climate change, painting a hopeful picture of the future.

4. AI in Art and Creativity:
Art and AI may seem like odd companions, but they are pushing the boundaries of creativity. This article will highlight the influence of AI in art and creativity, from creating original art pieces to enhancing human creativity. It will delve into the intriguing intersection of technology and art, showcasing the unexpected harmony between AI and human creativity.

5. The Future of Jobs with AI:
With the advent of AI, the job market is poised for a seismic shift. This article will delve into the implications of AI on the future of jobs, discussing the jobs AI will create and those it may render obsolete. It will provide a thought-provoking examination of how AI could redefine our workplaces and our understanding of work itself."""

@pytest.mark.vcr()
def test_crew_with_delegating_agents():
	tasks = [
		Task(
			description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
			agent=ceo
		)
	]

	crew = Crew(
		agents=[ceo, writer],
		process=Process.sequential,
		tasks=tasks,
	)

	assert crew.kickoff() == 'AI agents represent a significant stride in the evolution of artificial intelligence. These entities are designed to act autonomously, learn from their environment, and make decisions based on a set of predefined rules or through machine learning techniques. The potential of AI agents is tremendous and their versatility is unparalleled. They can be deployed in various sectors, from healthcare to finance, performing tasks with efficiency and accuracy that surpass human capabilities. In healthcare, AI agents can predict patient deterioration, offer personalized treatment suggestions, and manage patient flow. In finance, they can detect fraudulent transactions, manage investments, and provide personalized financial advice. Furthermore, the adaptability of AI agents allows them to learn and improve over time, becoming more proficient and effective in their tasks. This dynamic nature of AI agents is what sets them apart and posits them as a revolutionary force in the AI landscape. As we continue to explore and harness the capabilities of AI agents, we can expect them to play an increasingly integral role in shaping our world and the way we live.'

@pytest.mark.vcr()
def test_crew_verbose_output(capsys):
		tasks = [
				Task(
						description="Research AI advancements.",
						agent=researcher
				),
				Task(
						description="Write about AI in healthcare.",
						agent=writer
				)
		]

		crew = Crew(
				agents=[researcher, writer],
				tasks=tasks,
				process=Process.sequential,
				verbose=True
		)

		crew.kickoff()
		captured = capsys.readouterr()
		expected_strings = [
			"Working Agent: Researcher",
			"Starting Task: Research AI advancements. ...",
			"Task output:",
			"Working Agent: Senior Writer",
			"Starting Task: Write about AI in healthcare. ...",
			"Task output:"
		]

		for expected_string in expected_strings:
			assert expected_string in captured.out

		# Now test with verbose set to False
		crew.verbose = False
		crew.kickoff()
		captured = capsys.readouterr()
		assert captured.out == ""
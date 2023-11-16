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

	assert crew.kickoff() == """1. AI and Ethics: In a world that is increasingly being dominated by Artificial Intelligence, the question of ethics is ever more important. This article will delve into the complex intersection of AI and ethics, exploring how the decisions made by AI can impact society. From privacy concerns to the accountability of AI decisions, this piece will provide readers with a comprehensive understanding of the ethical dilemmas posed by AI.

2. The Role of AI in Climate Change: Climate change is the defining issue of our time. This article will examine how AI is playing a pivotal role in combating this global challenge. From predicting climate patterns to optimizing renewable energy use, the piece will highlight how AI is not just a part of the problem, but also a crucial part of the solution.

3. AI in Healthcare: The fusion of AI and healthcare holds a transformative potential. This article will explore how AI is revolutionizing healthcare, from improving diagnosis and treatment to enhancing patient care and hospital management. It will show how AI is not just improving healthcare outcomes but also driving a more efficient and patient-centered healthcare system.

4. The Future of AI and Work: The rise of AI has sparked a lively debate on its impact on jobs and the future of work. This article will explore this topic in-depth, examining both the potential job losses and the new opportunities created by AI. It will provide a balanced and insightful analysis of how AI is reshaping the world of work.

5. AI in Space Exploration: The final frontier is not beyond the reach of AI. This article will spotlight the role of AI in space exploration, from analyzing vast amounts of astronomical data to autonomous spacecraft navigating the vast expanse of space. The piece will highlight how AI is not just aiding but also accelerating our quest to explore the universe."""

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

	assert crew.kickoff() == 'The Senior Writer produced an amazing paragraph about AI Agents: "Artificial Intelligence (AI) agents, the cutting-edge technology that is reshaping the digital landscape, are software entities that autonomously perform tasks to achieve specific goals. These agents, programmed to make decisions based on their environment, are the driving force behind a multitude of innovations, from self-driving cars to personalized recommendations in e-commerce. They are pushing boundaries in various sectors, mitigating human error, increasing efficiency, and revolutionizing customer experience. The importance of AI agents is underscored by their ability to adapt and learn, ushering in a new era of technology where machines can mimic, and often surpass, human intelligence. Understanding AI agents is akin to peering into the future, a future where technology is seamless, intuitive, and astoundingly smart."'

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
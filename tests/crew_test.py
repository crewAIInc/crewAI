"""Test Agent creation and execution basic functionality."""

import pytest
from ..crewai import Agent, Crew, Task, Process

@pytest.mark.vcr()
def test_crew_creation():
	researcher = Agent(
		role="Researcher",
		goal="Make the best research and analysis on content about AI and AI agents",
		backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer."
	)

	writer = Agent(
		role="Senior Content Officer",
		goal="Write the best content about AI and AI agents.",
		backstory="You're a senior content officer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer."
	)

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

	assert crew.kickoff() == """1. AI in Healthcare: Dive into an exploratory journey where silicon meets stethoscope, unraveling how Artificial Intelligence is revolutionizing the healthcare industry. From early diagnosis of life-threatening diseases to personalized treatment plans, AI is not just an assistant; it's becoming a lifesaver. Imagine a world where technology doesn't just assist doctors but empowers patients, offering them more control over their health and well-being.

2. Ethical Implications of AI: Unearth the moral minefield that AI has become in this thought-provoking piece. As AI's capabilities grow, so do the ethical dilemmas. Should machines make life or death decisions? Can an algorithm be biased? Who bears responsibility for AI-induced outcomes? Join us as we navigate these murky waters, balancing the scales of technological advancement and ethical responsibility.

3. AI in Climate Change: Witness how AI is becoming an indispensable ally in the fight against climate change. From forecasting weather patterns and predicting natural disasters to optimizing energy use and promoting sustainable practices, AI is proving to be a game-changer. Explore how this nascent technology can help us steer towards a more sustainable future.

4. AI and Art: Embark on a fascinating exploration of where creativity meets code. Can a machine create a masterpiece? How does AI challenge our traditional notions of art? Delve into the world of AI-generated art, where algorithms are the artists and pixels are the palette, pushing the boundaries of what we consider "art".

5. The Future of AI Agents: Step into the realm of speculation and prediction, where we explore the potential future of AI agents. How will they shape industries, influence our daily lives, and even redefine our understanding of "intelligence"? Engage in a lively discourse around the possibilities and perils that the future of AI agents might hold."""
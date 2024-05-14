from crewai.agent import Agent
from crewai.utilities import i18n

class PlanningManagerAgent:
	def __init__(self, llm, verbose):
		self.agent = Agent(
									role=i18n.retrieve("planning_manager_agent", "role"),
									goal=i18n.retrieve("planning_manager_agent", "goal"),
									backstory=i18n.retrieve("planning_manager_agent", "backstory"),
									verbose=verbose,
									llm=llm,
								)
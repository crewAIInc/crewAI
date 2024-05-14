from crewai.agent import Agent
from crewai.tools.agent_tools import AgentTools
from crewai.utilities import I18N

class HierarchicalManagerAgent:
	def __init__(self, llm, agents, verbose):
		i18n = I18N()
		self.agent = Agent(
									role=i18n.retrieve("hierarchical_manager_agent", "role"),
									goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
									backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
									tools=AgentTools(agents=agents).tools(),
									llm=llm,
									verbose=verbose,
								)
from crewai.agents.agent_builder.utilities.base_agent_tool import BaseAgentTools


class AgentTools(BaseAgentTools):
    """Default tools around agent delegation"""

    def tools(self):
        from langchain.tools import StructuredTool

        coworkers = ", ".join([f"{agent.role}" for agent in self.agents])
        tools = [
            StructuredTool.from_function(
                func=self.delegate_work,
                name="Delegate work to coworker",
                description=self.i18n.tools("delegate_work").format(
                    coworkers=coworkers
                ),
            ),
            StructuredTool.from_function(
                func=self.ask_question,
                name="Ask question to coworker",
                description=self.i18n.tools("ask_question").format(coworkers=coworkers),
            ),
        ]
        return tools

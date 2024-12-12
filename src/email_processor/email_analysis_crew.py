"""
Email analysis crew implementation using CrewAI.
Handles comprehensive email analysis including thread history and sender research.
"""
from crewai import Agent, Task, Crew, Process
from typing import Dict, List, Optional
from datetime import datetime

from .gmail_tool import GmailTool

class EmailAnalysisCrew:
    """
    Crew for analyzing emails and determining response strategy.
    """

    def __init__(self, gmail_tool: Optional[GmailTool] = None):
        """Initialize analysis crew with required tools"""
        self.gmail_tool = gmail_tool or GmailTool()
        self._create_agents()

    def _create_agents(self):
        """Create specialized agents for email analysis"""
        self.context_analyzer = Agent(
            role="Email Context Analyst",
            name="Context Analyzer",
            goal="Analyze email context and history",
            backstory="Expert at understanding email threads and communication patterns",
            tools=[self.gmail_tool],
            verbose=True
        )

        self.research_specialist = Agent(
            role="Research Specialist",
            name="Research Expert",
            goal="Research sender and company background",
            backstory="Skilled at gathering and analyzing business and personal information",
            tools=[self.gmail_tool],
            verbose=True
        )

        self.response_strategist = Agent(
            role="Response Strategist",
            name="Strategy Expert",
            goal="Determine optimal response approach",
            backstory="Expert at developing communication strategies",
            tools=[self.gmail_tool],
            verbose=True
        )

    def analyze_email(self,
                     email: Dict,
                     thread_history: List[Dict],
                     sender_info: Dict) -> Dict:
        """
        Analyze email with comprehensive context.

        Args:
            email: Current email data
            thread_history: Previous thread messages
            sender_info: Information about the sender

        Returns:
            Dict: Analysis results including response decision
        """
        try:
            # Create analysis crew
            crew = Crew(
                agents=[
                    self.context_analyzer,
                    self.research_specialist,
                    self.response_strategist
                ],
                tasks=[
                    Task(
                        description="Analyze email context and thread history",
                        agent=self.context_analyzer
                    ),
                    Task(
                        description="Research sender and company background",
                        agent=self.research_specialist
                    ),
                    Task(
                        description="Determine response strategy",
                        agent=self.response_strategist
                    )
                ],
                verbose=True
            )

            # Execute analysis
            results = crew.kickoff()

            # Process results
            return {
                "email_id": email["id"],
                "thread_id": email["thread_id"],
                "response_needed": results[-1].get("response_needed", False),
                "priority": results[-1].get("priority", "low"),
                "similar_threads": results[0].get("similar_threads", []),
                "sender_context": results[1].get("sender_context", {}),
                "company_info": results[1].get("company_info", {}),
                "response_strategy": results[-1].get("strategy", {})
            }

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {
                "email_id": email.get("id", "unknown"),
                "thread_id": email.get("thread_id", "unknown"),
                "error": f"Analysis failed: {str(e)}",
                "response_needed": False,
                "priority": "error"
            }

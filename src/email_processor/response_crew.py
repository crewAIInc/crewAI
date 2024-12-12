"""
Response crew implementation for email processing.
Handles response generation with comprehensive context.
"""
from crewai import Agent, Task, Crew, Process
from typing import Dict, List, Optional
from datetime import datetime

from .gmail_tool import GmailTool

class ResponseCrew:
    """
    Crew for drafting email responses based on analysis.
    """

    def __init__(self, gmail_tool: Optional[GmailTool] = None):
        """Initialize response crew with required tools"""
        self.gmail_tool = gmail_tool or GmailTool()
        self._create_agents()

    def _create_agents(self):
        """Create specialized agents for response generation"""
        self.content_strategist = Agent(
            role="Content Strategist",
            name="Strategy Expert",
            goal="Develop response content strategy",
            backstory="Expert at planning effective communication approaches",
            tools=[self.gmail_tool],
            verbose=True
        )

        self.response_writer = Agent(
            role="Email Writer",
            name="Content Creator",
            goal="Write effective email responses",
            backstory="Skilled at crafting clear and impactful email content",
            tools=[self.gmail_tool],
            verbose=True
        )

        self.quality_reviewer = Agent(
            role="Quality Reviewer",
            name="Content Reviewer",
            goal="Ensure response quality and appropriateness",
            backstory="Expert at reviewing and improving email communications",
            tools=[self.gmail_tool],
            verbose=True
        )

    def draft_response(self,
                      email: Dict,
                      analysis: Dict,
                      thread_history: List[Dict]) -> Dict:
        """
        Generate response using comprehensive context.

        Args:
            email: Current email data
            analysis: Analysis results for the email
            thread_history: Previous messages in the thread

        Returns:
            Dict: Generated response data
        """
        try:
            # Create response crew
            crew = Crew(
                agents=[
                    self.content_strategist,
                    self.response_writer,
                    self.quality_reviewer
                ],
                tasks=[
                    Task(
                        description="Develop response strategy",
                        agent=self.content_strategist
                    ),
                    Task(
                        description="Write email response",
                        agent=self.response_writer
                    ),
                    Task(
                        description="Review and improve response",
                        agent=self.quality_reviewer
                    )
                ],
                verbose=True
            )

            # Generate response
            results = crew.kickoff()

            # Process results
            return {
                "email_id": email["id"],
                "thread_id": email["thread_id"],
                "response_text": results[-1].get("response_text", ""),
                "strategy": results[0],
                "context_used": {
                    "analysis": analysis,
                    "thread_size": len(thread_history)
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "reviewed": True,
                    "review_feedback": results[-1].get("feedback", {})
                }
            }

        except Exception as e:
            print(f"Response generation error: {str(e)}")
            return {
                "email_id": email.get("id", "unknown"),
                "thread_id": email.get("thread_id", "unknown"),
                "error": f"Response generation failed: {str(e)}",
                "response_text": None,
                "strategy": None
            }

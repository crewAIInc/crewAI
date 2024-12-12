"""
Email processing flow implementation using CrewAI.
Handles email polling, analysis, and response generation.
"""
from crewai import Flow
from typing import Dict, List, Optional
from datetime import datetime

from .models import EmailState
from .gmail_tool import GmailTool
from .email_analysis_crew import EmailAnalysisCrew
from .response_crew import ResponseCrew

class EmailProcessingFlow(Flow):
    """
    Flow for processing emails using CrewAI.
    Implements email fetching, analysis, and response generation.
    """

    def __init__(self, gmail_tool: Optional[GmailTool] = None):
        """Initialize flow with required tools and crews"""
        super().__init__()
        self.gmail_tool = gmail_tool or GmailTool()
        self.analysis_crew = EmailAnalysisCrew(gmail_tool=self.gmail_tool)
        self.response_crew = ResponseCrew(gmail_tool=self.gmail_tool)
        self._state = EmailState()
        self._initialize_state()

    def _initialize_state(self):
        """Initialize flow state attributes"""
        if not hasattr(self._state, "latest_emails"):
            self._state.latest_emails = []
        if not hasattr(self._state, "analysis_results"):
            self._state.analysis_results = {}
        if not hasattr(self._state, "generated_responses"):
            self._state.generated_responses = {}
        if not hasattr(self._state, "errors"):
            self._state.errors = {}

    def kickoff(self) -> Dict:
        """
        Execute the email processing flow.

        Returns:
            Dict: Flow execution results
        """
        try:
            # Fetch latest emails (limited to 5)
            self._state.latest_emails = self.gmail_tool.get_latest_emails(limit=5)

            # Analyze each email
            for email in self._state.latest_emails:
                email_id = email.get('id')
                thread_id = email.get('thread_id')
                sender_email = email.get('sender')  # Now matches test format

                analysis = self.analysis_crew.analyze_email(
                    email=email,
                    thread_history=self._get_thread_history(thread_id),
                    sender_info=self._get_sender_info(sender_email)
                )

                self._state.analysis_results[email_id] = analysis

                # Generate response if needed
                if analysis.get("response_needed", False):
                    response = self.response_crew.draft_response(
                        email=email,
                        analysis=analysis,
                        thread_history=self._get_thread_history(thread_id)
                    )
                    self._state.generated_responses[email_id] = response

            return {
                "emails_processed": len(self._state.latest_emails),
                "analyses_completed": len(self._state.analysis_results),
                "responses_generated": len(self._state.generated_responses)
            }

        except Exception as e:
            error_data = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self._state.errors["flow_execution"] = [error_data]  # Changed to list for multiple errors
            return {"error": error_data}

    def _get_thread_history(self, thread_id: str) -> List[Dict]:
        """Get thread history for email context"""
        try:
            return self.gmail_tool.get_thread_history(thread_id)
        except Exception:
            return []

    def _get_sender_info(self, sender_email: str) -> Dict:
        """Get sender information for context"""
        try:
            return self.gmail_tool.get_sender_info(sender_email)
        except Exception:
            return {}

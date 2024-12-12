"""
Test suite for email processing flow implementation.
"""
import pytest
from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock
from crewai.tools import BaseTool

from email_processor.models import EmailState
from email_processor.email_flow import EmailProcessingFlow
from email_processor.email_analysis_crew import EmailAnalysisCrew
from email_processor.response_crew import ResponseCrew

class MockGmailTool(BaseTool):
    """Mock Gmail tool for testing"""
    name: str = "Gmail Tool"
    description: str = "Tool for interacting with Gmail"

    def get_latest_emails(self, limit: int = 5) -> List[Dict]:
        """Mock getting latest emails"""
        return [
            {
                "id": f"email_{i}",
                "thread_id": f"thread_{i}",
                "subject": f"Test Email {i}",
                "sender": "test@example.com",
                "body": f"Test email body {i}",
                "date": datetime.now().isoformat()
            }
            for i in range(limit)
        ]

    def get_thread_history(self, thread_id: str) -> List[Dict]:
        """Mock getting thread history"""
        return [
            {
                "id": f"history_{i}",
                "thread_id": thread_id,
                "subject": f"Previous Email {i}",
                "sender": "test@example.com",
                "body": f"Previous email body {i}",
                "date": datetime.now().isoformat()
            }
            for i in range(3)
        ]

    def get_sender_info(self, email: str) -> Dict:
        """Mock getting sender information"""
        return {
            "email": email,
            "name": "Test User",
            "company": "Test Corp",
            "previous_threads": ["thread_1", "thread_2"],
            "interaction_history": {
                "total_emails": 10,
                "last_interaction": datetime.now().isoformat()
            }
        }

    def _run(self, method: str = "get_latest_emails", **kwargs) -> Dict:
        """Required implementation of BaseTool._run"""
        if method == "get_latest_emails":
            return self.get_latest_emails(kwargs.get("limit", 5))
        elif method == "get_thread_history":
            return self.get_thread_history(kwargs.get("thread_id"))
        elif method == "get_sender_info":
            return self.get_sender_info(kwargs.get("email"))
        return None

@pytest.fixture
def mock_crews(monkeypatch):
    """Mock analysis and response crews"""
    def mock_analyze_email(*args, **kwargs):
        email = kwargs.get("email", {})
        return {
            "email_id": email.get("id", "unknown"),
            "thread_id": email.get("thread_id", "unknown"),
            "response_needed": True,
            "priority": "high",
            "similar_threads": ["thread_1"],
            "sender_context": {"previous_interactions": 5},
            "company_info": {"name": "Test Corp", "industry": "Technology"},
            "response_strategy": {"tone": "professional", "key_points": ["previous collaboration"]}
        }

    def mock_draft_response(*args, **kwargs):
        email = kwargs.get("email", {})
        return {
            "email_id": email.get("id", "unknown"),
            "response_text": "Thank you for your email. We appreciate your continued collaboration.",
            "strategy": {"type": "professional", "focus": "relationship building"},
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "reviewed": True,
                "review_feedback": {"quality": "high", "tone": "appropriate"}
            }
        }

    monkeypatch.setattr(EmailAnalysisCrew, "analyze_email", mock_analyze_email)
    monkeypatch.setattr(ResponseCrew, "draft_response", mock_draft_response)

@pytest.fixture
def email_flow(monkeypatch):
    """Create email flow with mocked components"""
    mock_tool = MockGmailTool()
    def mock_init(self):
        self.gmail_tool = mock_tool
        self.analysis_crew = EmailAnalysisCrew(gmail_tool=mock_tool)
        self.response_crew = ResponseCrew(gmail_tool=mock_tool)
        self._state = EmailState()
        self._initialize_state()

    monkeypatch.setattr(EmailProcessingFlow, "__init__", mock_init)
    return EmailProcessingFlow()

def test_email_flow_initialization(email_flow):
    """Test flow initialization and state setup"""
    # Verify state initialization
    assert hasattr(email_flow._state, "latest_emails")
    assert hasattr(email_flow._state, "analysis_results")
    assert hasattr(email_flow._state, "generated_responses")
    assert isinstance(email_flow._state.latest_emails, list)
    assert isinstance(email_flow._state.analysis_results, dict)
    assert isinstance(email_flow._state.generated_responses, dict)

def test_email_fetching(email_flow):
    """Test email fetching with 5-email limit"""
    email_flow.kickoff()

    # Verify email fetching
    assert len(email_flow._state.latest_emails) <= 5
    assert len(email_flow._state.latest_emails) > 0
    assert all(isinstance(email, dict) for email in email_flow._state.latest_emails)

def test_email_analysis(email_flow, mock_crews):
    """Test email analysis and response decision"""
    email_flow.kickoff()

    # Verify analysis results
    assert len(email_flow._state.analysis_results) > 0
    for email_id, analysis in email_flow._state.analysis_results.items():
        assert "response_needed" in analysis
        assert "priority" in analysis
        assert isinstance(analysis["response_needed"], bool)

def test_response_generation(email_flow, mock_crews):
    """Test response generation for emails needing response"""
    email_flow.kickoff()

    # Verify response generation
    for email_id, analysis in email_flow._state.analysis_results.items():
        if analysis["response_needed"]:
            assert email_id in email_flow._state.generated_responses
            response = email_flow._state.generated_responses[email_id]
            assert "response_text" in response
            assert "strategy" in response
            assert "metadata" in response

def test_complete_flow(email_flow, mock_crews):
    """Test complete email processing flow"""
    result = email_flow.kickoff()

    # Verify complete flow execution
    assert len(email_flow._state.latest_emails) <= 5
    assert isinstance(email_flow._state.analysis_results, dict)
    assert isinstance(email_flow._state.generated_responses, dict)

    # Verify response generation for emails needing response
    for email_id, analysis in email_flow._state.analysis_results.items():
        if analysis["response_needed"]:
            assert email_id in email_flow._state.generated_responses
            assert email_flow._state.generated_responses[email_id]["email_id"] == email_id

def test_error_handling(email_flow):
    """Test error handling in flow execution"""
    # Simulate error in email fetching by modifying _run method
    original_run = email_flow.gmail_tool._run

    def mock_run(method: str = None, **kwargs):
        if method == "get_latest_emails":
            raise Exception("Test error")
        return original_run(method, **kwargs)

    email_flow.gmail_tool._run = mock_run
    result = email_flow.kickoff()

    # Verify error handling
    assert "flow_execution" in email_flow._state.errors
    assert isinstance(email_flow._state.errors["flow_execution"], list)
    assert len(email_flow._state.errors["flow_execution"]) > 0
    assert "Test error" in email_flow._state.errors["flow_execution"][0]["error"]

    # Restore original method
    email_flow.gmail_tool._run = original_run

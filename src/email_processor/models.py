"""
Models for email processing state management using Pydantic.
Handles state for email analysis, thread history, and response generation.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class SenderInfo(BaseModel):
    """Information about email senders and their companies"""
    name: str
    email: str
    company: Optional[str] = None
    title: Optional[str] = None
    company_info: Optional[Dict] = Field(default_factory=dict)
    interaction_history: List[Dict] = Field(default_factory=list)
    last_interaction: Optional[datetime] = None
    notes: Optional[str] = None

class EmailThread(BaseModel):
    """Email thread information and history"""
    thread_id: str
    subject: str
    participants: List[str]
    messages: List[Dict] = Field(default_factory=list)
    last_update: datetime
    labels: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

class EmailAnalysis(BaseModel):
    """Analysis results for an email"""
    email_id: str
    sender_email: str
    importance: int = Field(ge=0, le=10)
    response_needed: bool = False
    response_deadline: Optional[datetime] = None
    similar_threads: List[str] = Field(default_factory=list)
    context_summary: Optional[str] = None
    action_items: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None

class EmailResponse(BaseModel):
    """Generated email response"""
    email_id: str
    thread_id: str
    response_text: str
    context_used: Dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
    approved: bool = False

class EmailState(BaseModel):
    """Main state container for email processing flow"""
    latest_emails: List[Dict] = Field(
        default_factory=list,
        description="Latest 5 emails fetched from Gmail"
    )
    thread_history: Dict[str, EmailThread] = Field(
        default_factory=dict,
        description="History of email threads indexed by thread_id"
    )
    sender_info: Dict[str, SenderInfo] = Field(
        default_factory=dict,
        description="Information about senders and their companies"
    )
    analysis_results: Dict[str, EmailAnalysis] = Field(
        default_factory=dict,
        description="Analysis results for processed emails"
    )
    response_decisions: Dict[str, bool] = Field(
        default_factory=dict,
        description="Decision whether to respond to each email"
    )
    generated_responses: Dict[str, EmailResponse] = Field(
        default_factory=dict,
        description="Generated responses for emails"
    )
    errors: Dict[str, Dict] = Field(
        default_factory=dict,
        description="Error information for flow execution"
    )

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields to be set

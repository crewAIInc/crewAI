"""Mock email data for testing email processing tool"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel

class MockEmailMessage(BaseModel):
    id: str
    from_email: str
    to: List[str]
    date: str
    subject: str
    body: str

class MockEmailThread(BaseModel):
    thread_id: str
    subject: str
    messages: List[MockEmailMessage]

    def dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "messages": [msg.dict() for msg in self.messages]
        }

class MockSenderInfo(BaseModel):
    name: str
    company: str
    previous_threads: List[str]
    last_interaction: str
    interaction_frequency: str

MOCK_EMAILS = {
    "thread_1": MockEmailThread(
        thread_id="thread_1",
        subject="Meeting Follow-up",
        messages=[
            MockEmailMessage(
                id="msg1",
                from_email="john@example.com",
                to=["user@company.com"],
                date=(datetime.now() - timedelta(days=2)).isoformat(),
                subject="Meeting Follow-up",
                body="Thanks for the great discussion yesterday. Looking forward to next steps."
            ),
            MockEmailMessage(
                id="msg2",
                from_email="user@company.com",
                to=["john@example.com"],
                date=(datetime.now() - timedelta(days=1)).isoformat(),
                subject="Re: Meeting Follow-up",
                body="Great meeting indeed. I'll prepare the proposal by next week."
            )
        ]
    ),
    "thread_2": MockEmailThread(
        thread_id="thread_2",
        subject="Project Proposal",
        messages=[
            MockEmailMessage(
                id="msg3",
                from_email="john@example.com",
                to=["user@company.com"],
                date=(datetime.now() - timedelta(days=30)).isoformat(),
                subject="Project Proposal",
                body="Here's the initial project proposal for your review."
            )
        ]
    ),
    "thread_3": MockEmailThread(
        thread_id="thread_3",
        subject="Quick Question",
        messages=[
            MockEmailMessage(
                id="msg4",
                from_email="sarah@othercompany.com",
                to=["user@company.com"],
                date=datetime.now().isoformat(),
                subject="Quick Question",
                body="Do you have time for a quick call tomorrow?"
            )
        ]
    )
}

MOCK_SENDERS = {
    "john@example.com": MockSenderInfo(
        name="John Smith",
        company="Example Corp",
        previous_threads=["thread_1", "thread_2"],
        last_interaction=(datetime.now() - timedelta(days=1)).isoformat(),
        interaction_frequency="weekly"
    ),
    "sarah@othercompany.com": MockSenderInfo(
        name="Sarah Johnson",
        company="Other Company Ltd",
        previous_threads=["thread_3"],
        last_interaction=datetime.now().isoformat(),
        interaction_frequency="first_time"
    )
}

def get_mock_thread(thread_id: str) -> Optional[MockEmailThread]:
    return MOCK_EMAILS.get(thread_id)

def get_mock_sender_info(email: str) -> Optional[MockSenderInfo]:
    return MOCK_SENDERS.get(email)

def find_similar_threads(query: str) -> List[MockEmailThread]:
    similar = []
    query = query.lower()
    for thread in MOCK_EMAILS.values():
        if (query in thread.subject.lower() or
            any(query in msg.body.lower() for msg in thread.messages)):
            similar.append(thread)
    return similar

def get_sender_threads(sender_email: str) -> List[MockEmailThread]:
    sender = MOCK_SENDERS.get(sender_email)
    if not sender:
        return []
    return [MOCK_EMAILS[thread_id] for thread_id in sender.previous_threads if thread_id in MOCK_EMAILS]

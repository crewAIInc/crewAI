"""Email processing tool for CrewAI"""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime
from .mock_email_data import (
    get_mock_thread,
    get_mock_sender_info,
    find_similar_threads,
    get_sender_threads
)

class EmailMessage(BaseModel):
    from_email: str
    to: List[str]
    date: str
    subject: str
    body: str

class EmailThread(BaseModel):
    thread_id: str
    messages: List[EmailMessage]
    subject: str
    participants: List[str]
    last_message_date: datetime

class EmailTool(BaseTool):
    name: str = "Email Processing Tool"
    description: str = "Processes emails, finds similar threads, and analyzes communication history"

    def _run(self, operation: Literal["get_thread", "find_similar", "get_history", "analyze_context"],
             thread_id: Optional[str] = None,
             query: Optional[str] = None,
             sender_email: Optional[str] = None,
             max_results: int = 10) -> Dict:
        if operation == "get_thread":
            if not thread_id:
                raise ValueError("thread_id is required for get_thread operation")
            return self.get_email_thread(thread_id).dict()

        elif operation == "find_similar":
            if not query:
                raise ValueError("query is required for find_similar operation")
            threads = self.find_similar_threads(query, max_results)
            return {"threads": [thread.dict() for thread in threads]}

        elif operation == "get_history":
            if not sender_email:
                raise ValueError("sender_email is required for get_history operation")
            return self.get_sender_history(sender_email)

        elif operation == "analyze_context":
            if not thread_id:
                raise ValueError("thread_id is required for analyze_context operation")
            return self.analyze_thread_context(thread_id)

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_email_thread(self, thread_id: str) -> EmailThread:
        mock_thread = get_mock_thread(thread_id)
        if not mock_thread:
            raise Exception(f"Thread not found: {thread_id}")

        messages = [
            EmailMessage(
                from_email=msg.from_email,
                to=msg.to,
                date=msg.date,
                subject=msg.subject,
                body=msg.body
            ) for msg in mock_thread.messages
        ]

        return EmailThread(
            thread_id=mock_thread.thread_id,
            messages=messages,
            subject=mock_thread.subject,
            participants=list({msg.from_email for msg in mock_thread.messages} |
                           {to for msg in mock_thread.messages for to in msg.to}),
            last_message_date=datetime.fromisoformat(mock_thread.messages[-1].date)
        )

    def find_similar_threads(self, query: str, max_results: int = 10) -> List[EmailThread]:
        similar_mock_threads = find_similar_threads(query)[:max_results]
        return [
            EmailThread(
                thread_id=thread.thread_id,
                messages=[
                    EmailMessage(
                        from_email=msg.from_email,
                        to=msg.to,
                        date=msg.date,
                        subject=msg.subject,
                        body=msg.body
                    ) for msg in thread.messages
                ],
                subject=thread.subject,
                participants=list({msg.from_email for msg in thread.messages} |
                               {to for msg in thread.messages for to in msg.to}),
                last_message_date=datetime.fromisoformat(thread.messages[-1].date)
            )
            for thread in similar_mock_threads
        ]

    def get_sender_history(self, sender_email: str) -> Dict:
        sender_info = get_mock_sender_info(sender_email)
        if not sender_info:
            return {
                "name": "",
                "company": "",
                "threads": [],
                "last_interaction": None,
                "interaction_frequency": "none"
            }

        sender_threads = get_sender_threads(sender_email)
        return {
            "name": sender_info.name,
            "company": sender_info.company,
            "threads": [
                self.get_email_thread(thread.thread_id).dict()
                for thread in sender_threads
            ],
            "last_interaction": sender_info.last_interaction,
            "interaction_frequency": sender_info.interaction_frequency
        }

    def analyze_thread_context(self, thread_id: str) -> Dict:
        try:
            # Get thread data
            thread = self.get_email_thread(thread_id)

            # Get sender info from first message
            sender_email = thread.messages[0].from_email
            sender_info = self.get_sender_history(sender_email)

            # Find similar threads
            similar_threads = self.find_similar_threads(thread.subject)

            # Create context summary
            context_summary = {
                "thread_length": len(thread.messages),
                "thread_duration": (
                    datetime.fromisoformat(thread.messages[-1].date) -
                    datetime.fromisoformat(thread.messages[0].date)
                ).days,
                "participant_count": len(thread.participants),
                "has_previous_threads": len(similar_threads) > 0
            }

            return {
                "thread": thread.dict(),
                "sender_info": sender_info,
                "similar_threads": [t.dict() for t in similar_threads],
                "context_summary": context_summary
            }

        except Exception as e:
            print(f"Error analyzing thread context: {str(e)}")
            return {
                "thread": {},
                "sender_info": {},
                "similar_threads": [],
                "context_summary": {
                    "thread_length": 0,
                    "thread_duration": 0,
                    "participant_count": 0,
                    "has_previous_threads": False
                }
            }

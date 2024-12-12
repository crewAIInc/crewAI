"""Mock LLM implementation for testing CrewAI agents"""
from typing import Dict, Any, Optional
import json

class MockLLM:
    """Mock LLM responses for testing agent interactions"""

    @staticmethod
    def analyze_context(input_data: Dict) -> str:
        """Generate mock analysis of email context"""
        thread = input_data.get("thread", {})
        similar_threads = input_data.get("similar_threads", [])
        context_summary = input_data.get("context_summary", {})

        context = {
            "thread_type": "business" if "meeting" in thread.get("subject", "").lower() else "general",
            "conversation_stage": "follow_up" if context_summary.get("thread_length", 0) > 1 else "initial",
            "sender_relationship": "established" if context_summary.get("thread_length", 0) > 2 else "new",
            "urgency_indicators": any(word in thread.get("subject", "").lower() for word in ["urgent", "asap", "quick"]),
            "key_topics": ["meeting", "proposal"] if "meeting" in thread.get("subject", "").lower() else ["general inquiry"],
            "thread_summary": {
                "subject": thread.get("subject", "Unknown"),
                "message_count": context_summary.get("thread_length", 0),
                "participant_count": context_summary.get("participant_count", 0),
                "has_previous_threads": context_summary.get("has_previous_threads", False)
            }
        }
        return json.dumps(context)

    @staticmethod
    def research_sender(input_data: Dict) -> str:
        """Generate mock research about sender"""
        sender_info = input_data.get("sender_info", {})
        return json.dumps({
            "sender_background": f"{sender_info.get('name', 'Unknown')} at {sender_info.get('company', 'Unknown Company')}",
            "company_info": "Leading technology firm" if "Corp" in sender_info.get("company", "") else "Emerging business",
            "interaction_history": {
                "frequency": sender_info.get("interaction_frequency", "new"),
                "last_contact": sender_info.get("last_interaction", "unknown"),
                "thread_count": len(sender_info.get("previous_threads", []))
            }
        })

    @staticmethod
    def determine_response(input_data: Dict) -> str:
        """Generate mock response strategy determination"""
        context = input_data.get("context", {})
        research = input_data.get("research", {})
        thread_data = input_data.get("thread_data", {}).get("thread", {})
        interaction_frequency = input_data.get("interaction_frequency", "new")
        similar_threads_count = input_data.get("similar_threads_count", 0)

        # Determine if response is needed based on multiple factors
        is_business = context.get("thread_type") == "business"
        is_urgent = context.get("urgency_indicators", False)
        is_frequent = interaction_frequency == "weekly"
        has_previous_threads = similar_threads_count > 0
        is_first_time = interaction_frequency == "first_time"

        # Response needed logic - more conservative for first-time senders
        response_needed = (
            (is_urgent and (not is_first_time or is_business)) or  # Urgent only matters if not first-time or is business
            (is_business and not is_first_time) or  # Business communications need response if not first-time
            is_frequent or  # Regular contacts need response
            (has_previous_threads and context.get("conversation_stage") == "follow_up")  # Follow-ups to existing threads
        )

        # Priority level determination - more nuanced
        priority_level = "high" if (
            (is_urgent and (is_business or is_frequent)) or  # Urgent is only high priority for business or frequent contacts
            (is_business and is_frequent)  # Regular business contacts are high priority
        ) else "medium" if (
            is_business or  # Business communications are at least medium priority
            (has_previous_threads and is_frequent) or  # Regular contacts with history are medium priority
            (is_urgent and is_first_time)  # First-time urgent requests are medium priority
        ) else "low"  # Everything else is low priority

        return json.dumps({
            "response_needed": response_needed,
            "priority_level": priority_level,
            "response_strategy": {
                "tone": "professional" if is_business else "casual",
                "key_points": [
                    "Acknowledge previous interaction" if has_previous_threads else "Introduce context",
                    "Address specific inquiries",
                    "Propose next steps" if priority_level in ["high", "medium"] else "Maintain relationship"
                ],
                "timing": "urgent" if priority_level == "high" else "standard",
                "considerations": [
                    f"Relationship: {'established' if is_frequent else 'new'}",
                    f"Previous threads: {similar_threads_count}",
                    f"Business context: {research.get('company_info', '')}",
                    f"Interaction frequency: {interaction_frequency}",
                    f"Priority level: {priority_level}",
                    f"Response needed: {response_needed}",
                    f"First time sender: {is_first_time}"
                ]
            }
        })

    @staticmethod
    def create_content_strategy(input_data: Dict) -> str:
        """Generate mock content strategy for email response"""
        thread_context = input_data.get("thread_context", {})
        analysis = input_data.get("analysis", {})

        return json.dumps({
            "tone": "professional",
            "key_points": [
                "Reference previous interaction",
                "Address main topics",
                "Propose next steps"
            ],
            "structure": {
                "greeting": "Personalized based on relationship",
                "context": "Reference previous messages",
                "main_content": "Address key points",
                "closing": "Action-oriented"
            },
            "considerations": [
                f"Relationship: {analysis.get('priority', 'medium')}",
                "Previous communication history",
                "Business context"
            ]
        })

    @staticmethod
    def draft_response(input_data: Dict) -> str:
        """Generate mock email response draft"""
        strategy = input_data.get("strategy", {})
        context = input_data.get("context", {})
        thread_data = input_data.get("thread_data", {})
        analysis = input_data.get("analysis", {})

        # Extract subject properly
        subject = thread_data.get("subject", "Unknown Subject")
        if isinstance(subject, str) and subject:
            subject = f"Re: {subject}"

        # Get context information
        priority = analysis.get("priority", "medium")
        sender_info = analysis.get("analysis", {}).get("research", {})
        context_info = analysis.get("analysis", {}).get("context", {})
        is_business = context_info.get("thread_type") == "business"
        is_followup = context_info.get("conversation_stage") == "follow_up"

        # Generate appropriate greeting
        sender_name = sender_info.get('sender_background', '[Contact]').split(' at ')[0]
        greeting = f"Dear {sender_name}" if is_business else f"Hi {sender_name}"

        # Generate appropriate context line
        context_line = (
            "Thank you for your follow-up regarding" if is_followup
            else "Thank you for reaching out about"
        )

        # Generate appropriate priority/context acknowledgment
        priority_line = (
            "I understand the urgency of your request and will address it promptly" if priority == "high"
            else "I appreciate you bringing this to my attention and will address your points"
        )

        # Generate appropriate closing
        closing = (
            "I look forward to our continued collaboration" if is_followup
            else "I look forward to discussing this further"
        )

        content = f"""
{greeting},

{context_line} {subject.replace('Re: ', '')}.

{priority_line}.

{strategy.get('key_points', [''])[0]}
{strategy.get('key_points', ['', ''])[1]}
{strategy.get('key_points', ['', '', ''])[2]}

{closing}.

Best regards,
[Your Name]
        """.strip()

        return json.dumps({
            "subject": subject,
            "content": content,
            "tone_used": strategy.get("tone", "professional"),
            "points_addressed": strategy.get("key_points", []),
            "draft_version": "1.0",
            "context_used": {
                "priority": priority,
                "relationship": context_info.get("sender_relationship", "professional"),
                "background": sender_info.get("sender_background", "")
            }
        })

    @staticmethod
    def review_response(input_data: Dict) -> str:
        """Generate mock review of email response"""
        draft = input_data.get("draft", {})
        strategy = input_data.get("strategy", {})
        context = input_data.get("context", {})

        content = draft.get("content", "")
        points_addressed = draft.get("points_addressed", [])
        context_used = draft.get("context_used", {})

        suggestions = []
        if "[Contact]" in content:
            suggestions.append("Add specific contact name")
        if "[Your Name]" in content:
            suggestions.append("Add sender name")
        if len(points_addressed) < len(strategy.get("key_points", [])):
            suggestions.append("Address all key points")

        return json.dumps({
            "final_content": content,
            "subject": draft.get("subject", ""),
            "review_notes": {
                "tone_appropriate": True,
                "points_addressed": len(suggestions) == 0,
                "clarity": "High",
                "professionalism": "Maintained",
                "context_awareness": {
                    "priority_reflected": context_used.get("priority") in content.lower(),
                    "relationship_acknowledged": context_used.get("relationship") in content.lower(),
                    "background_used": bool(context_used.get("background"))
                }
            },
            "suggestions_implemented": suggestions if suggestions else [
                "All requirements met",
                "Context appropriately used",
                "Professional tone maintained"
            ],
            "version": "1.1"
        })

def mock_agent_executor(agent_role: str, task_input: Dict) -> str:
    """Mock agent execution with predefined responses"""
    llm = MockLLM()

    if "Context Analyzer" in agent_role:
        return llm.analyze_context(task_input)
    elif "Research Specialist" in agent_role:
        return llm.research_sender(task_input)
    elif "Response Strategist" in agent_role:
        return llm.determine_response(task_input)
    elif "Content Strategist" in agent_role:
        return llm.create_content_strategy(task_input)
    elif "Response Writer" in agent_role:
        return llm.draft_response(task_input)
    elif "Quality Reviewer" in agent_role:
        return llm.review_response(task_input)

    return json.dumps({"error": "Unknown agent role"})

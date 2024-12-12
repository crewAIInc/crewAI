"""Test script for complete email processing workflow"""
from email_analysis_crew import EmailAnalysisCrew
from response_crew import ResponseCrew
from email_tool import EmailTool
from mock_email_data import MockEmailThread
import json

def test_email_analysis(email_tool, analysis_crew, thread_id):
    """Test comprehensive email analysis including similar threads and research"""
    print("\nAnalyzing email thread...")

    # Get thread context
    thread = email_tool.get_email_thread(thread_id)
    print(f"\nThread subject: {thread.subject}")

    # Find similar threads
    similar = email_tool.find_similar_threads(thread.subject)
    print(f"\nFound {len(similar)} similar threads")

    # Get sender history
    sender = thread.messages[0].from_email
    sender_info = email_tool.get_sender_history(sender)
    print(f"\nSender: {sender_info['name']} from {sender_info['company']}")
    print(f"Previous interactions: {sender_info['interaction_frequency']}")

    # Analyze with crew
    analysis_result = analysis_crew.analyze_email(thread_id)
    print(f"\nAnalysis Results:")
    print(f"Response needed: {analysis_result.get('response_needed', False)}")
    print(f"Priority: {analysis_result.get('priority', 'error')}")
    print(f"Decision factors:")
    context = analysis_result.get('analysis', {}).get('context', {})
    print(f"- Thread type: {context.get('thread_type', 'unknown')}")
    print(f"- Similar threads found: {analysis_result.get('similar_threads_found', 0)}")
    print(f"- Interaction frequency: {sender_info.get('interaction_frequency', 'unknown')}")
    print(f"- Urgency indicators: {context.get('urgency_indicators', False)}")
    print(f"- Conversation stage: {context.get('conversation_stage', 'unknown')}")

    return analysis_result

def test_complete_workflow():
    """Test the complete email processing workflow"""
    try:
        print("\nTesting Complete Email Processing Workflow")
        print("=========================================")

        # Initialize tools and crews
        email_tool = EmailTool()
        analysis_crew = EmailAnalysisCrew()
        response_crew = ResponseCrew()

        # Test 1: Process email requiring response (weekly interaction)
        print("\nTest 1: Processing email requiring response")
        print("------------------------------------------")
        thread_id = "thread_1"  # Meeting follow-up thread from frequent contact

        analysis_result = test_email_analysis(email_tool, analysis_crew, thread_id)

        if analysis_result.get('response_needed', False):
            print("\nGenerating response...")
            response_result = response_crew.draft_response(thread_id, analysis_result)
            print("\nGenerated Response:")
            print(json.dumps(response_result.get('response', {}), indent=2))

            # Verify response matches context
            print("\nResponse Analysis:")
            print(f"Tone matches relationship: {response_result['response']['review_notes']['context_awareness']['relationship_acknowledged']}")
            print(f"Priority reflected: {response_result['response']['review_notes']['context_awareness']['priority_reflected']}")
            print(f"Background used: {response_result['response']['review_notes']['context_awareness']['background_used']}")
        else:
            print("\nNo response required.")

        # Test 2: Process email not requiring response (first-time sender)
        print("\nTest 2: Processing email not requiring response")
        print("----------------------------------------------")
        thread_id = "thread_3"  # First-time contact

        analysis_result = test_email_analysis(email_tool, analysis_crew, thread_id)

        if analysis_result.get('response_needed', False):
            print("\nGenerating response...")
            response_result = response_crew.draft_response(thread_id, analysis_result)
            print("\nGenerated Response:")
            print(json.dumps(response_result.get('response', {}), indent=2))

            # Verify response matches context
            print("\nResponse Analysis:")
            context = analysis_result.get('analysis', {}).get('context', {})
            print(f"Thread type: {context.get('thread_type', 'unknown')}")
            print(f"Conversation stage: {context.get('conversation_stage', 'unknown')}")
            print(f"Response priority: {analysis_result.get('priority', 'unknown')}")
        else:
            print("\nNo response required - First time sender with no urgent context")

        print("\nWorkflow test completed successfully!")
        return True

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return False

if __name__ == "__main__":
    test_complete_workflow()

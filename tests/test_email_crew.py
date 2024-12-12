"""Test script for email analysis crew"""
from email_analysis_crew import EmailAnalysisCrew
from email_tool import EmailTool
import json

def test_email_tool():
    """Test the email tool functionality first"""
    try:
        tool = EmailTool()

        # Test get_thread operation
        result = tool._run("get_thread", thread_id="thread_1")
        print("\nThread details:")
        print(f"Subject: {result['subject']}")
        print(f"Participants: {', '.join(result['participants'])}")

        # Test find_similar operation
        result = tool._run("find_similar", query="meeting")
        print("\nSimilar threads:")
        for thread in result['threads']:
            print(f"- {thread['subject']}")

        # Test get_history operation
        result = tool._run("get_history", sender_email="john@example.com")
        print("\nSender history:")
        print(f"Name: {result['name']}")
        print(f"Company: {result['company']}")

        # Test analyze_context operation
        result = tool._run("analyze_context", thread_id="thread_1")
        print("\nContext analysis:")
        print(f"Thread length: {result['context_summary']['thread_length']}")
        print(f"Relationship: {result['context_summary']['sender_relationship']}")

        return True
    except Exception as e:
        print(f"Tool test error: {str(e)}")
        return False

def test_email_analysis():
    """Test the email analysis crew functionality"""
    if not test_email_tool():
        print("Skipping crew test due to tool failure")
        return False

    try:
        # Initialize crew
        crew = EmailAnalysisCrew()
        print("\nTesting email analysis crew...")

        # Test analysis of thread_1 (meeting follow-up thread)
        print("\nAnalyzing meeting follow-up thread...")
        result = crew.analyze_email("thread_1")

        print("\nAnalysis Results:")
        print(f"Thread ID: {result['thread_id']}")
        print(f"Response Needed: {result['response_needed']}")
        print(f"Priority: {result['priority']}")

        if result['response_needed']:
            print("\nContext Analysis:")
            print(json.dumps(result['analysis']['context'], indent=2))
            print("\nSender Research:")
            print(json.dumps(result['analysis']['research'], indent=2))
            print("\nResponse Strategy:")
            print(json.dumps(result['analysis']['strategy'], indent=2))

        # Test analysis of thread_3 (new inquiry)
        print("\nAnalyzing new inquiry thread...")
        result = crew.analyze_email("thread_3")

        print("\nAnalysis Results:")
        print(f"Thread ID: {result['thread_id']}")
        print(f"Response Needed: {result['response_needed']}")
        print(f"Priority: {result['priority']}")

        if result['response_needed']:
            print("\nContext Analysis:")
            print(json.dumps(result['analysis']['context'], indent=2))
            print("\nSender Research:")
            print(json.dumps(result['analysis']['research'], indent=2))
            print("\nResponse Strategy:")
            print(json.dumps(result['analysis']['strategy'], indent=2))

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_email_analysis()

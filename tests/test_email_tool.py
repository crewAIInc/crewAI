"""Test script for email processing tool"""
from gmail_tool import EmailTool
from datetime import datetime, timedelta

def test_email_tool():
    """Test email processing tool functionality"""
    try:
        # Initialize tool
        email_tool = EmailTool()

        # Test getting email thread
        print("\nTesting thread retrieval...")
        thread = email_tool.get_email_thread("thread_1")
        print(f"Retrieved thread: {thread.subject}")
        print(f"Participants: {', '.join(thread.participants)}")
        print(f"Messages: {len(thread.messages)}")

        # Test finding similar threads
        print("\nTesting similar thread search...")
        similar = email_tool.find_similar_threads("meeting")
        print(f"Found {len(similar)} similar threads")
        for t in similar:
            print(f"- {t.subject}")

        # Test sender history
        print("\nTesting sender history...")
        history = email_tool.get_sender_history("john@example.com")
        print(f"Sender: {history['name']} from {history['company']}")
        print(f"Last interaction: {history['last_interaction']}")
        print(f"Interaction frequency: {history['interaction_frequency']}")

        # Test thread context analysis
        print("\nTesting thread context analysis...")
        context = email_tool.analyze_thread_context("thread_1")
        print("Context Summary:")
        print(f"Thread length: {context['context_summary']['thread_length']} messages")
        print(f"Time span: {context['context_summary']['time_span']} days")
        print(f"Relationship: {context['context_summary']['sender_relationship']}")

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_email_tool()

"""Test script for response crew functionality"""
from response_crew import ResponseCrew
from email_analysis_crew import EmailAnalysisCrew
import json

def test_response_crew():
    """Test the response crew functionality"""
    try:
        # First get analysis results
        analysis_crew = EmailAnalysisCrew()
        analysis_result = analysis_crew.analyze_email("thread_1")

        if not analysis_result.get("response_needed", False):
            print("No response needed for this thread")
            return True

        # Initialize response crew
        response_crew = ResponseCrew()
        print("\nTesting response crew...")

        # Draft response
        result = response_crew.draft_response("thread_1", analysis_result)

        print("\nResponse Results:")
        print(f"Thread ID: {result['thread_id']}")

        if result.get("error"):
            print(f"Error: {result['error']}")
            return False

        print("\nContent Strategy:")
        print(json.dumps(result['strategy_used'], indent=2))

        print("\nFinal Response:")
        print(json.dumps(result['response'], indent=2))

        print("\nAll tests completed successfully!")
        return True

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_response_crew()

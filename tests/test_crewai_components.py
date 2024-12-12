from crewai import Agent, Task, Crew, Process
from typing import List
import json
from pydantic import BaseModel
import os

# Define output models
class EmailAnalysis(BaseModel):
    needs_response: bool
    priority: str
    context: str

# Sample email data for testing
SAMPLE_EMAIL = {
    "subject": "Meeting Follow-up",
    "body": "Thanks for the great discussion yesterday. Looking forward to next steps.",
    "sender": "john@example.com"
}

# Test Agent Creation
researcher = Agent(
    role="Email Researcher",
    goal="Analyze email content and gather relevant context",
    backstory="Expert at analyzing communication patterns and gathering contextual information",
    verbose=True,
    allow_delegation=True
)

# Test Task Creation
analysis_task = Task(
    description=f"Analyze this email content and determine if it requires a response: {json.dumps(SAMPLE_EMAIL)}",
    agent=researcher,
    expected_output="Detailed analysis of email content and response requirement",
    output_json=EmailAnalysis
)

# Test Crew Creation with Sequential Process
crew = Crew(
    agents=[researcher],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)

# Test execution with error handling
if __name__ == "__main__":
    try:
        # Ensure we have an API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Please set OPENAI_API_KEY environment variable")
            exit(1)

        result = crew.kickoff()
        print("Execution Results:", result)

        # Access structured output
        if hasattr(result, "output") and result.output:
            analysis = EmailAnalysis.parse_raw(result.output)
            print("\nStructured Analysis:")
            print(f"Needs Response: {analysis.needs_response}")
            print(f"Priority: {analysis.priority}")
            print(f"Context: {analysis.context}")
    except Exception as e:
        print(f"Error during execution: {str(e)}")

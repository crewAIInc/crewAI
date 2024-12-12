# Email Processing System with CrewAI

A smart email processing system that uses CrewAI to analyze Gmail messages and automatically generate appropriate responses based on context and history.

## Quick Start

```bash
# Install required packages
pip install crewai 'crewai[tools]'
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Example Usage

```python
from email_analysis_crew import EmailAnalysisCrew
from response_crew import ResponseCrew
from gmail_tool import GmailTool

# Initialize Gmail connection
gmail = GmailTool()
gmail.authenticate()

# Create analysis crew
analysis_crew = EmailAnalysisCrew(
    gmail_tool=gmail,
    config={
        "check_similar_threads": True,
        "analyze_sender_history": True
    }
)

# Process new emails
new_emails = gmail.get_new_emails()
for email in new_emails:
    # Analyze email and decide on response
    analysis = analysis_crew.analyze_email(email)

    # If analysis determines response is needed
    if analysis["response_needed"]:
        # Create new crew for response generation
        response_crew = ResponseCrew(
            email_context=analysis,
            gmail_tool=gmail
        )

        # Generate and send response
        response = response_crew.generate_response()
        gmail.send_response(response)
```

## How It Works

1. **Email Analysis**
   - System connects to Gmail
   - Retrieves new emails
   - Analyzes email context and history
   - Checks for similar threads
   - Researches sender and company

2. **Response Decision**
   The analysis crew decides whether to respond based on:
   - Email urgency
   - Sender relationship
   - Previous interactions
   - Business context
   - Similar thread history

3. **Response Generation**
   If a response is needed:
   - New response crew is created
   - Analyzes email context
   - Generates appropriate response
   - Reviews for tone and content
   - Sends through Gmail

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install crewai 'crewai[tools]'
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
   ```

2. **Gmail Setup**
   - Visit [Google Cloud Console](https://console.cloud.google.com)
   - Create new project
   - Enable Gmail API
   - Create OAuth credentials
   - Download as `credentials.json`

3. **Configure Authentication**
   ```python
   from gmail_tool import GmailTool

   gmail = GmailTool()
   gmail.authenticate()  # Opens browser for auth
   ```

## Configuration

```python
# Analysis configuration
config = {
    "check_similar_threads": True,    # Look for similar conversations
    "analyze_sender_history": True,   # Check previous interactions
    "research_company": True,         # Research sender's company
    "priority_threshold": 0.7         # Threshold for response
}

# Create crews
analysis_crew = EmailAnalysisCrew(
    gmail_tool=gmail,
    config=config
)

response_crew = ResponseCrew(
    email_context=analysis,
    gmail_tool=gmail
)
```

## Components

- `EmailAnalysisCrew`: Analyzes emails and makes response decisions
- `ResponseCrew`: Generates appropriate responses
- `GmailTool`: Handles Gmail integration
- `EmailTool`: Provides email processing utilities

## Error Handling

The system includes handling for:
- Gmail API connection issues
- Authentication errors
- Rate limiting
- Invalid email formats

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## License

MIT License

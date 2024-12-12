# Email Processor

A CrewAI-powered email processing system that analyzes and responds to emails intelligently.

## Features

- Automated email analysis using AI agents
- Smart response generation based on context
- Thread history analysis
- Sender research and profiling
- Priority-based response handling

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/email-processor.git
cd email-processor

# Install using UV package manager (recommended)
uv sync --dev --all-extras
uv build
pip install dist/*.whl

# Or install using pip
pip install -r requirements.txt
python setup.py install
```

## Dependencies

- Python 3.8+
- CrewAI
- Google API Python Client (for Gmail integration)

## Quick Start

```python
from email_processor import EmailAnalysisCrew, ResponseCrew, GmailTool

# Initialize the email analysis crew
analysis_crew = EmailAnalysisCrew()

# Analyze an email thread
analysis = analysis_crew.analyze_email("thread_id")

# If response is needed, use the response crew
if analysis["response_needed"]:
    response_crew = ResponseCrew()
    response = response_crew.draft_response("thread_id", analysis)
    print(f"Generated response: {response['response']['content']}")
```

## Gmail Integration Setup

1. Enable Gmail API in Google Cloud Console
2. Create OAuth 2.0 credentials
3. Download credentials file
4. Set up authentication:

```python
from email_processor import GmailTool

gmail_tool = GmailTool()
# Follow authentication prompts
```

## Components

### EmailAnalysisCrew
Analyzes email threads and determines response strategy:
```python
analysis = analysis_crew.analyze_email(thread_id)
print(f"Response needed: {analysis['response_needed']}")
print(f"Priority: {analysis['priority']}")
```

### ResponseCrew
Generates contextually appropriate email responses:
```python
response = response_crew.draft_response(thread_id, analysis)
print(f"Response: {response['response']['content']}")
```

### GmailTool
Handles Gmail API integration:
```python
gmail_tool = GmailTool()
new_threads = gmail_tool.get_new_threads()
```

### EmailTool
Core email processing functionality:
```python
email_tool = EmailTool()
thread_context = email_tool.analyze_thread_context(thread_id)
```

## Complete Example

```python
from email_processor import EmailAnalysisCrew, ResponseCrew, GmailTool

def process_new_emails():
    # Initialize components
    gmail_tool = GmailTool()
    analysis_crew = EmailAnalysisCrew()
    response_crew = ResponseCrew()

    # Process new emails
    new_threads = gmail_tool.get_new_threads()
    for thread_id in new_threads:
        # Analyze thread
        analysis = analysis_crew.analyze_email(thread_id)

        print(f"Thread {thread_id}:")
        print(f"Priority: {analysis['priority']}")
        print(f"Similar threads found: {analysis['similar_threads_found']}")

        # Generate response if needed
        if analysis["response_needed"]:
            response = response_crew.draft_response(thread_id, analysis)
            print(f"Response generated: {response['response']['content']}")

if __name__ == "__main__":
    process_new_emails()
```

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/email-processor.git

# Install development dependencies
cd email-processor
uv sync --dev --all-extras

# Run tests
python -m pytest tests/
```

## Project Structure

```
email_processor/
├── src/
│   └── email_processor/
│       ├── __init__.py
│       ├── email_analysis_crew.py
│       ├── response_crew.py
│       ├── email_tool.py
│       ├── gmail_tool.py
│       ├── gmail_auth.py
│       └── mock_email_data.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

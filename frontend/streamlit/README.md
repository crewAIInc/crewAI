# CrewAI Frontend - Streamlit Prototype (Phase 1)

This is the Phase 1 implementation of the CrewAI frontend using Streamlit. It provides a basic but functional interface for managing CrewAI's core components: Crews, Agents, and Tasks.

## Features

- View all crews and their associated agents
- Filter and view tasks by crew
- Real-time status overview with key metrics
- Basic error handling demonstration

## Project Structure

```
frontend/streamlit/
├── crewai_streamlit.py    # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
streamlit run crewai_streamlit.py
```

The application will start and open in your default web browser.

## Mock Data

The prototype uses mock data to demonstrate functionality. In a production environment, this would be replaced with actual API calls to the CrewAI backend.

## Next Steps

This prototype serves as a foundation for the more advanced Phase 2 implementation using Next.js. Key learnings and user feedback from this phase will inform the development of the full-featured frontend.

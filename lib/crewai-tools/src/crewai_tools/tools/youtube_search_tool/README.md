# YouTubeSearchTool

A tool for searching YouTube videos using the YouTube Data API v3.

## Features

- Search YouTube videos by keyword/query
- Returns structured results with title, video ID, URL, description, and publish date
- Configurable number of results (1-50)
- Uses official Google YouTube Data API v3
- Proper error handling and type hints

## Installation

```bash
# Install with YouTube API dependencies
pip install "crewai-tools[youtube]"

# Or install dependencies manually
pip install google-api-python-client>=2.100.0 google-auth>=2.25.0
```

## Configuration

You need a YouTube Data API v3 key from Google Cloud Console:

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project or select existing one
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Set the environment variable:

```bash
export YOUTUBE_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from crewai_tools.tools.youtube_search_tool.youtube_search_tool import YouTubeSearchTool

# Initialize the tool
tool = YouTubeSearchTool()

# Search for videos
results = tool._run(search_query="Python programming tutorial", max_results=5)

# Results is a list of dicts:
# [
#     {
#         "title": "Python Tutorial for Beginners",
#         "video_id": "abc123",
#         "url": "https://www.youtube.com/watch?v=abc123",
#         "description": "Learn Python in this comprehensive tutorial...",
#         "published_at": "2024-01-15T10:30:00Z"
#     },
#     ...
# ]
```

### Using with CrewAI Agents

```python
from crewai import Agent, Task, Crew
from crewai_tools.tools.youtube_search_tool.youtube_search_tool import YouTubeSearchTool

youtube_search = YouTubeSearchTool()

researcher = Agent(
    role="YouTube Researcher",
    goal="Find relevant YouTube videos on given topics",
    backstory="You are an expert at finding high-quality educational content on YouTube.",
    tools=[youtube_search],
    verbose=True,
)

search_task = Task(
    description="Search for YouTube videos about 'Python async programming' and return the top 5 results",
    expected_output="A list of 5 YouTube videos with title, URL, and description for each.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[search_task], verbose=True)
result = crew.kickoff()
```

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `search_query` | str | Yes | - | The search query to use |
| `max_results` | int | No | 5 | Maximum number of results (1-50) |

## Output Format

Returns a list of dictionaries with the following keys:

- `title` (str): Video title
- `video_id` (str): YouTube video ID
- `url` (str): Full YouTube URL
- `description` (str): Video description
- `published_at` (str): ISO 8601 publish timestamp

## Error Handling

The tool raises appropriate exceptions:

- `ValueError`: If `YOUTUBE_API_KEY` environment variable is not set
- `ImportError`: If `google-api-python-client` is not installed
- `RuntimeError`: If the YouTube API request fails (with detailed error message)

## Requirements

- Python 3.10+
- google-api-python-client >= 2.100.0
- google-auth >= 2.25.0
- Valid YouTube Data API v3 key

## API Quota

The YouTube Data API v3 has a quota system:
- Search costs 100 units per request
- Default quota: 10,000 units/day
- Maximum results per request: 50

## License

MIT
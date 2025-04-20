# Enhanced Templating with Jinja2

CrewAI now supports enhanced templating using Jinja2, while maintaining compatibility with the existing templating system.

## Basic Usage

The basic templating syntax remains the same:

```python
from crewai import Agent, Task, Crew

# Define inputs
inputs = {
    "topic": "Artificial Intelligence",
    "year": 2024,
    "count": 5
}

# Create an agent with template variables
researcher = Agent(
    role="{topic} Researcher",
    goal="Research the latest developments in {topic} for {year}",
    backstory="You're an expert in {topic} with years of experience"
)

# Create a task with template variables
research_task = Task(
    description="Research {topic} and provide {count} key insights",
    expected_output="A list of {count} key insights about {topic} in {year}",
    agent=researcher
)

# Create a crew and pass inputs
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    inputs=inputs
)

# Run the crew
result = crew.kickoff()
```

## Advanced Features

The new templating system adds support for container types, object attributes, conditional statements, loops, and filters:

### Container Types

```python
inputs = {
    "topics": ["AI", "Machine Learning", "Data Science"],
    "details": {"main_theme": "Technology Trends", "subtopics": ["Ethics", "Applications"]}
}

# Access list items
task = Task(
    description="Research {{topics[0]}} and {{topics[1]}}",
    expected_output="Analysis of the topics"
)

# Access dictionary items
task = Task(
    description="Research {{details.main_theme}} with focus on {{details.subtopics[0]}}",
    expected_output="Detailed analysis"
)
```

### Conditional Statements

```python
inputs = {
    "topic": "AI",
    "priority": "high",
    "deadline": "2024-12-31"
}

task = Task(
    description="{% if priority == 'high' %}URGENT: {% endif %}Research {topic}{% if deadline %} by {{deadline}}{% endif %}",
    expected_output="A report on {topic}"
)
```

### Loop Statements

```python
inputs = {
    "topics": ["AI", "Machine Learning", "Data Science"]
}

task = Task(
    description="Research the following topics: {% for topic in topics %}{{topic}}{% if not loop.last %}, {% endif %}{% endfor %}",
    expected_output="A report covering multiple topics"
)
```

### Filters

```python
from datetime import datetime

inputs = {
    "topic": "AI",
    "date": datetime.now()
}

task = Task(
    description="Research {topic} as of {{date|date('%Y-%m-%d')}}",
    expected_output="A report on {topic}"
)
```

### Custom Objects

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    
    def __str__(self):
        return f"{self.name} ({self.age})"

inputs = {
    "author": Person(name="John Doe", age=35)
}

task = Task(
    description="Write a report authored by {author}",
    expected_output="A report by {{author.name}}"
)
```

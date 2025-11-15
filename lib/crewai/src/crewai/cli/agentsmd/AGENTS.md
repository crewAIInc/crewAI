# CrewAI Development Guidelines

## Project Context
This is a CrewAI project for building AI agent systems with multi-agent collaboration capabilities.

## Core Principles
- You are a Software Engineer who specializes in GenAI and Python
- You are an expert on CrewAI
- Use Python 3 as the primary programming language
- Use type hints consistently
- Use Pandas for data manipulation and analysis
- Optimize for readability over premature optimization
- Write modular code, using separate files for models, data loading, training, and evaluation
- Follow PEP8 style guide for Python code
- Never use relative imports, always use absolute imports
- Configure pre-commit hooks to automatically format and lint code before committing.
- Implement proper error handling and retry logic

## CrewAI Principles
- For ANY question about CrewAI, use the comprehensive documentation available at[CrewAI Documentation](https://docs.crewai.com/llms-full.txt)
- Use uv as a build system and `pyproject.toml` for managing dependencies and build settings.
- When creating new crews or flows, use the cli commands:
  - `crewai create crew <crew_name> --skip_provider`
  - `crewai create flow <flow_name> --skip_provider`
- When Creating Agents follow the guides here [Crafting Effective Agents](https://docs.crewai.com/guides/agents/crafting-effective-agents)
- Agents should have clear, single responsibilities
- Tasks should be atomic and well-defined
- Use delegation between agents when appropriate
- Prioritize classes over functions for better organization and maintainability.

## Dependencies
Check latest version of CrewAI Framework on [PyPi.org](https://pypi.org/project/crewai/)

```txt
crewai>=0.186.1
crewai-tools>=0.71.0
```

## Code Organization

### Directory Structure For Crews

```txt
  my_crew/
  ├── .gitignore
  ├── knowledge/: directory where you can store knowledge base files (e.g. pdfs, docs, etc.)
  ├── pyproject.toml: project dependencies
  ├── README.md: project README
  ├── .env: environment variables, including secrets
  ├── tests/: directory for storing tests
  └── src/
      └── my_crew/
          ├── main.py: entry point for the application and configuration to run crews
          ├── crew.py: crew definition file, using CrewBase class to define agents and tasks
          ├── tools/: directory for storing tools
          │   ├── custom_tool.py: example tool, use it as a template to generate tools
          └── config/
              ├── agents.yaml: agent definitions
              └── tasks.yaml: task definitions
```

### Directory Structure For Flows

```txt
  my_flow/
  ├── .gitignore
  ├── pyproject.toml: project dependencies
  ├── README.md: project README
  ├── .env: environment variables, including secrets
  ├── tests/: directory for storing tests
  └── src/
      └── my_flow/
          ├── main.py: entry point for the application and flow methods
          ├── crews/: crews that are used in the flow
          │   └── poem_crew/: crew that is used in the flow
          │       ├── config/
          │       │   ├── agents.yaml: agent definitions
          │       │   └── tasks.yaml: task definitions
          │       ├── poem_crew.py: crew definition file, using CrewBase class to define agents and tasks
          │       └── other crew files and directories as needed
          └── tools/: tools that are used in the flow
              └── custom_tool.py: example tool, use it as a templpate to generate tools
```

### File Naming Conventions
  - Use descriptive, lowercase names with underscores (e.g., `weather_api_tool.py`).
  - Pydantic models should be singular (e.g., `ArticleSummary.py` -> `article_summary.py` and class `ArticleSummary`)

## Code Style and Patterns
- Maintain a modular architecture for flexibility and scalability.

### Agent Definition
- Create agents definition in the config/agents.yaml file
- Define agents with clear roles and backstories
- Use descriptive goal statements
- Implement verbose mode during development
- Use allow_delegation=True only when necessary

### Task Creation
- Create tasks definition in the config/tasks.yaml file
- Tasks should have clear, actionable descriptions
- Include expected output format
- Specify agent assignment explicitly
- Use context from other tasks when needed and configure it in tasks.yaml
- Leverage structured responses from LLM calls using Pydantic for output validation when necessary

### Tool Implementation

#### Using Built-in Tools
- Import from crewai_tools python package
- Configure with appropriate parameters
- Handle rate limits and API quotas
- Always learn what tools are available by using a command `ls .venv/lib/python3.12/site-packages/crewai_tools/*/**/**.py`

```python
from crewai_tools import SerperDevTool, WebsiteSearchTool

search_tool = SerperDevTool(
    api_key=os.getenv("SERPER_API_KEY"),
    n_results=5
)
```

#### Custom Tools
- Inherit from BaseTool
- Implement clear descriptions
- Handle errors gracefully

```python
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
```

### Crew Assembly
- Define clear process types (sequential, hierarchical, flow)
- Implement proper callbacks for monitoring
- Use memory when needed for context retention
- Use @CrewBase class to define agents and tasks and compose crew

### Flows

#### State Management

- Consider using `@persist` decorator to persist state for flows.
- Ensure that state is properly serialized and deserialized.

### Environment Variables
- Always use .env files for sensitive data
- Never commit API keys to version control
- Use os.getenv() with defaults
  
## Testing Patterns
- Place all tests in `tests/` directory

### Unit Testing Agents
```python
def test_agent_creation():
    agent = create_research_agent()
    assert agent.role == "Senior Research Analyst"
    assert agent.max_iter == 5
```

### Integration Testing Crews
```python
def test_crew_execution():
    crew = create_test_crew()
    result = crew.kickoff(inputs={"topic": "test"})
    assert result is not None
    assert "summary" in result
```

## Performance Considerations

### Memory Management
  - Be mindful of the memory footprint of agents and tasks.
  - Avoid storing large amounts of data in agent memory.
  - Release resources promptly after use.

### Caching Strategies
- Use embeddings for semantic search

## Security Best Practices

### Common Vulnerabilities
  - Prompt injection attacks.
  - Data breaches due to insecure storage of sensitive information.

### Input Validation
  - Validate all inputs from users and external sources.
  - Sanitize inputs to prevent code injection attacks.
  - Use regular expressions or validation libraries to enforce input constraints.
  
### Secure API Communication
  - Use HTTPS for all API communication.
  - Validate API responses to prevent data corruption.
  - Use secure authentication mechanisms for API access.

## Running & Deployment

- In order to run the crewai CLI command, always use the venv activated with `source .venv/bin/activate`.
- Run the crew locally with `crewai run`.
- Ensure that the user is logged in with `crewai login`.
- You can use `crewai org list` to list available organizations and `crewai org switch <id>` to switch to a specific organization.
- Here are available commands related to deployment:
  - `crewai deploy create` - create a new deployment
  - `crewai deploy list` - list all deployments
  - `crewai deploy remove -u <id>` - remove a deployment
  - `crewai deploy status -u <id>` - get the status of a deployment
  - `crewai deploy logs -u <id>` - get the logs of a deployment

## Common Pitfalls and Gotchas
- **Frequent Mistakes:**
  - Using overly complex prompts that are difficult to understand and maintain.
  - Failing to handle errors and exceptions gracefully.
  - Neglecting to validate inputs and outputs.
  - Not monitoring and logging application behavior.
- **Edge Cases:**
  - Handling unexpected LLM responses.
  - Dealing with rate limits and API errors.
  - Managing long-running tasks.
- **Version-Specific Issues:**
  - Consult the CrewAI changelog for information about new features and bug fixes.
- **Debugging Strategies:**
  - Use logging to track application behavior. Enable verbose=True during development.
  - Use a debugger to step through the code and inspect variables.
  - Use print statements to debug simple issues.

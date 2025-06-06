# A2A Protocol Integration

CrewAI supports the A2A (Agent-to-Agent) protocol, enabling your crews to participate in remote agent interoperability. This allows CrewAI crews to be exposed as remotely accessible agents that can communicate with other A2A-compatible systems.

## Overview

The A2A protocol is Google's standard for agent interoperability that enables bidirectional communication between agents. CrewAI's A2A integration provides:

- **Remote Interoperability**: Expose crews as A2A-compatible agents
- **Bidirectional Communication**: Enable full-duplex agent interactions
- **Protocol Compliance**: Full support for A2A specifications
- **Transport Flexibility**: Support for multiple transport protocols

## Installation

A2A support is available as an optional dependency:

```bash
pip install crewai[a2a]
```

## Basic Usage

### Creating an A2A Server

```python
from crewai import Agent, Crew, Task
from crewai.a2a import CrewAgentExecutor, start_a2a_server

# Create your crew
agent = Agent(
    role="Assistant",
    goal="Help users with their queries",
    backstory="A helpful AI assistant"
)

task = Task(
    description="Help with: {query}",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task])

# Create A2A executor
executor = CrewAgentExecutor(crew)

# Start A2A server
start_a2a_server(executor, host="0.0.0.0", port=10001)
```

### Custom Configuration

```python
from crewai.a2a import CrewAgentExecutor, create_a2a_app

# Create executor with custom content types
executor = CrewAgentExecutor(
    crew=crew,
    supported_content_types=['text', 'application/json', 'image/png']
)

# Create custom A2A app
app = create_a2a_app(
    executor,
    agent_name="My Research Crew",
    agent_description="A specialized research and analysis crew",
    transport="starlette"
)

# Run with custom ASGI server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Key Features

### CrewAgentExecutor

The `CrewAgentExecutor` class wraps CrewAI crews to implement the A2A `AgentExecutor` interface:

- **Asynchronous Execution**: Crews run asynchronously within the A2A protocol
- **Task Management**: Automatic handling of task lifecycle and cancellation
- **Error Handling**: Robust error handling with A2A-compliant responses
- **Output Conversion**: Automatic conversion of crew outputs to A2A artifacts

### Server Utilities

Convenience functions for starting A2A servers:

- `start_a2a_server()`: Quick server startup with default configuration
- `create_a2a_app()`: Create custom A2A applications for advanced use cases

## Protocol Compliance

CrewAI's A2A integration provides full protocol compliance:

- **Agent Cards**: Automatic generation of agent capability descriptions
- **Task Execution**: Asynchronous task processing with event queues
- **Artifact Management**: Conversion of crew outputs to A2A artifacts
- **Error Handling**: A2A-compliant error responses and status codes

## Use Cases

### Remote Agent Networks

Expose CrewAI crews as part of larger agent networks:

```python
# Multi-agent system with specialized crews
research_crew = create_research_crew()
analysis_crew = create_analysis_crew()
writing_crew = create_writing_crew()

# Expose each as A2A agents on different ports
start_a2a_server(CrewAgentExecutor(research_crew), port=10001)
start_a2a_server(CrewAgentExecutor(analysis_crew), port=10002)
start_a2a_server(CrewAgentExecutor(writing_crew), port=10003)
```

### Cross-Platform Integration

Enable CrewAI crews to work with other agent frameworks:

```python
# CrewAI crew accessible to other A2A-compatible systems
executor = CrewAgentExecutor(crew)
start_a2a_server(executor, host="0.0.0.0", port=10001)

# Other systems can now invoke this crew remotely
```

## Advanced Configuration

### Custom Agent Cards

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

# Custom agent card for specialized capabilities
agent_card = AgentCard(
    name="Specialized Research Crew",
    description="Advanced research and analysis capabilities",
    version="2.0.0",
    capabilities=AgentCapabilities(
        streaming=True,
        pushNotifications=False
    ),
    skills=[
        AgentSkill(
            id="research",
            name="Research Analysis",
            description="Comprehensive research and analysis",
            tags=["research", "analysis", "data"]
        )
    ]
)
```

### Error Handling

The A2A integration includes comprehensive error handling:

- **Validation Errors**: Input validation with clear error messages
- **Execution Errors**: Crew execution errors converted to A2A artifacts
- **Cancellation**: Proper task cancellation support
- **Timeouts**: Configurable timeout handling

## Best Practices

1. **Resource Management**: Monitor crew resource usage in server environments
2. **Error Handling**: Implement proper error handling in crew tasks
3. **Security**: Use appropriate authentication and authorization
4. **Monitoring**: Monitor A2A server performance and health
5. **Scaling**: Consider load balancing for high-traffic scenarios

## Limitations

- **Optional Dependency**: A2A support requires additional dependencies
- **Transport Support**: Currently supports Starlette transport only
- **Synchronous Crews**: Crews execute synchronously within async A2A context

## Examples

See the `examples/a2a_integration_example.py` file for a complete working example of A2A integration with CrewAI.

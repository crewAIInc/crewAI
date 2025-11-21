# BedrockInvokeAgentTool

The `BedrockInvokeAgentTool` enables CrewAI agents to invoke Amazon Bedrock Agents and leverage their capabilities within your workflows.

## Installation

```bash
pip install 'crewai[tools]'
```

## Requirements

- AWS credentials configured (either through environment variables or AWS CLI)
- `boto3` and `python-dotenv` packages
- Access to Amazon Bedrock Agents

## Usage

Here's how to use the tool with a CrewAI agent:

```python
from crewai import Agent, Task, Crew
from crewai_tools.aws.bedrock.agents.invoke_agent_tool import BedrockInvokeAgentTool

# Initialize the tool
agent_tool = BedrockInvokeAgentTool(
    agent_id="your-agent-id",
    agent_alias_id="your-agent-alias-id"
)

# Create a CrewAI agent that uses the tool
aws_expert = Agent(
    role='AWS Service Expert',
    goal='Help users understand AWS services and quotas',
    backstory='I am an expert in AWS services and can provide detailed information about them.',
    tools=[agent_tool],
    verbose=True
)

# Create a task for the agent
quota_task = Task(
    description="Find out the current service quotas for EC2 in us-west-2 and explain any recent changes.",
    agent=aws_expert
)

# Create a crew with the agent
crew = Crew(
    agents=[aws_expert],
    tasks=[quota_task],
    verbose=2
)

# Run the crew
result = crew.kickoff()
print(result)
```

## Tool Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| agent_id | str | Yes | None | The unique identifier of the Bedrock agent |
| agent_alias_id | str | Yes | None | The unique identifier of the agent alias |
| session_id | str | No | timestamp | The unique identifier of the session |
| enable_trace | bool | No | False | Whether to enable trace for debugging |
| end_session | bool | No | False | Whether to end the session after invocation |
| description | str | No | None | Custom description for the tool |

## Environment Variables

```bash
BEDROCK_AGENT_ID=your-agent-id           # Alternative to passing agent_id
BEDROCK_AGENT_ALIAS_ID=your-agent-alias-id # Alternative to passing agent_alias_id
AWS_REGION=your-aws-region               # Defaults to us-west-2
AWS_ACCESS_KEY_ID=your-access-key        # Required for AWS authentication
AWS_SECRET_ACCESS_KEY=your-secret-key    # Required for AWS authentication
```

## Advanced Usage

### Multi-Agent Workflow with Session Management

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools.aws.bedrock.agents.invoke_agent_tool import BedrockInvokeAgentTool

# Initialize tools with session management
initial_tool = BedrockInvokeAgentTool(
    agent_id="your-agent-id",
    agent_alias_id="your-agent-alias-id",
    session_id="custom-session-id"
)

followup_tool = BedrockInvokeAgentTool(
    agent_id="your-agent-id",
    agent_alias_id="your-agent-alias-id",
    session_id="custom-session-id"
)

final_tool = BedrockInvokeAgentTool(
    agent_id="your-agent-id",
    agent_alias_id="your-agent-alias-id",
    session_id="custom-session-id",
    end_session=True
)

# Create agents for different stages
researcher = Agent(
    role='AWS Service Researcher',
    goal='Gather information about AWS services',
    backstory='I am specialized in finding detailed AWS service information.',
    tools=[initial_tool]
)

analyst = Agent(
    role='Service Compatibility Analyst',
    goal='Analyze service compatibility and requirements',
    backstory='I analyze AWS services for compatibility and integration possibilities.',
    tools=[followup_tool]
)

summarizer = Agent(
    role='Technical Documentation Writer',
    goal='Create clear technical summaries',
    backstory='I specialize in creating clear, concise technical documentation.',
    tools=[final_tool]
)

# Create tasks
research_task = Task(
    description="Find all available AWS services in us-west-2 region.",
    agent=researcher
)

analysis_task = Task(
    description="Analyze which services support IPv6 and their implementation requirements.",
    agent=analyst
)

summary_task = Task(
    description="Create a summary of IPv6-compatible services and their key features.",
    agent=summarizer
)

# Create a crew with the agents and tasks
crew = Crew(
    agents=[researcher, analyst, summarizer],
    tasks=[research_task, analysis_task, summary_task],
    process=Process.sequential,
    verbose=2
)

# Run the crew
result = crew.kickoff()
```

## Use Cases

### Hybrid Multi-Agent Collaborations
- Create workflows where CrewAI agents collaborate with managed Bedrock agents running as services in AWS
- Enable scenarios where sensitive data processing happens within your AWS environment while other agents operate externally
- Bridge on-premises CrewAI agents with cloud-based Bedrock agents for distributed intelligence workflows

### Data Sovereignty and Compliance
- Keep data-sensitive agentic workflows within your AWS environment while allowing external CrewAI agents to orchestrate tasks
- Maintain compliance with data residency requirements by processing sensitive information only within your AWS account
- Enable secure multi-agent collaborations where some agents cannot access your organization's private data

### Seamless AWS Service Integration
- Access any AWS service through Amazon Bedrock Actions without writing complex integration code
- Enable CrewAI agents to interact with AWS services through natural language requests
- Leverage pre-built Bedrock agent capabilities to interact with AWS services like Bedrock Knowledge Bases, Lambda, and more

### Scalable Hybrid Agent Architectures
- Offload computationally intensive tasks to managed Bedrock agents while lightweight tasks run in CrewAI
- Scale agent processing by distributing workloads between local CrewAI agents and cloud-based Bedrock agents

### Cross-Organizational Agent Collaboration
- Enable secure collaboration between your organization's CrewAI agents and partner organizations' Bedrock agents
- Create workflows where external expertise from Bedrock agents can be incorporated without exposing sensitive data
- Build agent ecosystems that span organizational boundaries while maintaining security and data control
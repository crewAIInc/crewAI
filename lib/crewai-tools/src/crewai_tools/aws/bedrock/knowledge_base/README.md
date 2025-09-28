# BedrockKBRetrieverTool

The `BedrockKBRetrieverTool` enables CrewAI agents to retrieve information from Amazon Bedrock Knowledge Bases using natural language queries.

## Installation

```bash
pip install 'crewai[tools]'
```

## Requirements

- AWS credentials configured (either through environment variables or AWS CLI)
- `boto3` and `python-dotenv` packages
- Access to Amazon Bedrock Knowledge Base

## Usage

Here's how to use the tool with a CrewAI agent:

```python
from crewai import Agent, Task, Crew
from crewai_tools.aws.bedrock.knowledge_base.retriever_tool import BedrockKBRetrieverTool

# Initialize the tool
kb_tool = BedrockKBRetrieverTool(
    knowledge_base_id="your-kb-id",
    number_of_results=5
)

# Create a CrewAI agent that uses the tool
researcher = Agent(
    role='Knowledge Base Researcher',
    goal='Find information about company policies',
    backstory='I am a researcher specialized in retrieving and analyzing company documentation.',
    tools=[kb_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description="Find our company's remote work policy and summarize the key points.",
    agent=researcher
)

# Create a crew with the agent
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

# Run the crew
result = crew.kickoff()
print(result)
```

## Tool Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| knowledge_base_id | str | Yes | None | The unique identifier of the knowledge base (0-10 alphanumeric characters) |
| number_of_results | int | No | 5 | Maximum number of results to return |
| retrieval_configuration | dict | No | None | Custom configurations for the knowledge base query |
| guardrail_configuration | dict | No | None | Content filtering settings |
| next_token | str | No | None | Token for pagination |

## Environment Variables

```bash
BEDROCK_KB_ID=your-knowledge-base-id  # Alternative to passing knowledge_base_id
AWS_REGION=your-aws-region            # Defaults to us-east-1
AWS_ACCESS_KEY_ID=your-access-key     # Required for AWS authentication
AWS_SECRET_ACCESS_KEY=your-secret-key # Required for AWS authentication
```

## Response Format

The tool returns results in JSON format:

```json
{
  "results": [
    {
      "content": "Retrieved text content",
      "content_type": "text",
      "source_type": "S3",
      "source_uri": "s3://bucket/document.pdf",
      "score": 0.95,
      "metadata": {
        "additional": "metadata"
      }
    }
  ],
  "nextToken": "pagination-token",
  "guardrailAction": "NONE"
}
```

## Advanced Usage

### Custom Retrieval Configuration

```python
kb_tool = BedrockKBRetrieverTool(
    knowledge_base_id="your-kb-id",
    retrieval_configuration={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
            "overrideSearchType": "HYBRID"
        }
    }
)

policy_expert = Agent(
    role='Policy Expert',
    goal='Analyze company policies in detail',
    backstory='I am an expert in corporate policy analysis with deep knowledge of regulatory requirements.',
    tools=[kb_tool]
)
```

## Supported Data Sources

- Amazon S3
- Confluence
- Salesforce
- SharePoint
- Web pages
- Custom document locations
- Amazon Kendra
- SQL databases    

## Use Cases

### Enterprise Knowledge Integration
- Enable CrewAI agents to access your organization's proprietary knowledge without exposing sensitive data
- Allow agents to make decisions based on your company's specific policies, procedures, and documentation
- Create agents that can answer questions based on your internal documentation while maintaining data security

### Specialized Domain Knowledge
- Connect CrewAI agents to domain-specific knowledge bases (legal, medical, technical) without retraining models
- Leverage existing knowledge repositories that are already maintained in your AWS environment
- Combine CrewAI's reasoning with domain-specific information from your knowledge bases

### Data-Driven Decision Making
- Ground CrewAI agent responses in your actual company data rather than general knowledge
- Ensure agents provide recommendations based on your specific business context and documentation
- Reduce hallucinations by retrieving factual information from your knowledge bases

### Scalable Information Access
- Access terabytes of organizational knowledge without embedding it all into your models
- Dynamically query only the relevant information needed for specific tasks
- Leverage AWS's scalable infrastructure to handle large knowledge bases efficiently

### Compliance and Governance
- Ensure CrewAI agents provide responses that align with your company's approved documentation
- Create auditable trails of information sources used by your agents
- Maintain control over what information sources your agents can access
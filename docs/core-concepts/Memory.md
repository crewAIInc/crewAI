---
title: crewAI Memory Systems
description: Leveraging memory systems in the crewAI framework to enhance agent capabilities.
---

## Introduction to Memory Systems in crewAI
!!! note "Enhancing Agent Intelligence"
    The crewAI framework introduces a sophisticated memory system designed to significantly enhance the capabilities of AI agents. This system comprises short-term memory, long-term memory, entity memory, and newly identified contextual memory, each serving a unique purpose in aiding agents to remember, reason, and learn from past interactions.

## Memory System Components

| Component            | Description                                                  |
| :------------------- | :----------------------------------------------------------- |
| **Short-Term Memory**| Temporarily stores recent interactions and outcomes, enabling agents to recall and utilize information relevant to their current context. |
| **Long-Term Memory** | Preserves valuable insights and learnings from past executions, allowing agents to build and refine their knowledge over time. |
| **Entity Memory**    | Captures and organizes information about entities (people, places, concepts) encountered during tasks, facilitating deeper understanding and relationship mapping. |
| **Contextual Memory**| Maintains the context of interactions, aiding in the coherence and relevance of agent responses over a sequence of tasks or a conversation. |

## How Memory Systems Empower Agents

1. **Contextual Awareness**: With short-term and contextual memory, agents gain the ability to maintain context over a conversation or task sequence, leading to more coherent and relevant responses.

2. **Experience Accumulation**: Long-term memory allows agents to accumulate experiences, learning from past actions to improve future decision-making and problem-solving.

3. **Entity Understanding**: By maintaining entity memory, agents can recognize and remember key entities, enhancing their ability to process and interact with complex information.

## Implementing Memory in Your Crew

When configuring a crew, you can enable and customize each memory component to suit the crew's objectives and the nature of tasks it will perform.
By default, the memory system is disabled, and you can ensure it is active by setting `memory=True` in the crew configuration.
The memory will use OpenAI Embeddings by default, but you can change it by setting `embedder` to a different model.

### Example: Configuring Memory for a Crew

```python
from crewai import Crew, Agent, Task, Process

# Assemble your crew with memory capabilities
my_crew = Crew(
    agents=[...],
    tasks=[...],
    process=Process.sequential,
    memory=True,
    verbose=True
)
```

## Additional Embedding Providers

### Using OpenAI embeddings (already default)
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
				"provider": "openai",
				"config":{
						"model": 'text-embedding-3-small'
				}
		}
)
```

### Using Google AI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "google",
			"config":{
				"model": 'models/embedding-001',
				"task_type": "retrieval_document",
				"title": "Embeddings for Embedchain"
			}
		}
)
```

### Using Azure OpenAI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "azure_openai",
			"config":{
				"model": 'text-embedding-ada-002',
				"deployment_name": "you_embedding_model_deployment_name"
			}
		}
)
```

### Using GPT4ALL embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "gpt4all"
		}
)
```

### Using Vertex AI embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "vertexai",
			"config":{
				"model": 'textembedding-gecko'
			}
		}
)
```

### Using Cohere embeddings
```python
from crewai import Crew, Agent, Task, Process

my_crew = Crew(
		agents=[...],
		tasks=[...],
		process=Process.sequential,
		memory=True,
		verbose=True,
		embedder={
			"provider": "cohere",
			"config":{
				"model": "embed-english-v3.0"
    		"vector_dimension": 1024
			}
		}
)
```

## Benefits of Using crewAI's Memory System
- **Adaptive Learning:** Crews become more efficient over time, adapting to new information and refining their approach to tasks.
- **Enhanced Personalization:** Memory enables agents to remember user preferences and historical interactions, leading to personalized experiences.
- **Improved Problem Solving:** Access to a rich memory store aids agents in making more informed decisions, drawing on past learnings and contextual insights.

## Getting Started
Integrating crewAI's memory system into your projects is straightforward. By leveraging the provided memory components and configurations, you can quickly empower your agents with the ability to remember, reason, and learn from their interactions, unlocking new levels of intelligence and capability.
---
title: crewAI Memory Systems
description: Leveraging memory systems in the crewAI framework to enhance agent capabilities.
---

## Introduction to Memory Systems in crewAI
!!! note "Enhancing Agent Intelligence"
    The crewAI framework introduces a sophisticated memory system designed to significantly enhance the capabilities of AI agents. This system comprises short-term memory, long-term memory, entity memory, and contextual memory, each serving a unique purpose in aiding agents to remember, reason, and learn from past interactions.

## Memory System Components

| Component            | Description                                                  |
| :------------------- | :----------------------------------------------------------- |
| **Short-Term Memory**| Temporarily stores recent interactions and outcomes, enabling agents to recall and utilize information relevant to their current context during the current executions. |
| **Long-Term Memory** | Preserves valuable insights and learnings from past executions, allowing agents to build and refine their knowledge over time. So Agents can remember what they did right and wrong across multiple executions |
| **Entity Memory**    | Captures and organizes information about entities (people, places, concepts) encountered during tasks, facilitating deeper understanding and relationship mapping. |
| **Contextual Memory**| Maintains the context of interactions by combining `ShortTermMemory`, `LongTermMemory`, and `EntityMemory`, aiding in the coherence and relevance of agent responses over a sequence of tasks or a conversation. |

## How Memory Systems Empower Agents

1. **Contextual Awareness**: With short-term and contextual memory, agents gain the ability to maintain context over a conversation or task sequence, leading to more coherent and relevant responses.

2. **Experience Accumulation**: Long-term memory allows agents to accumulate experiences, learning from past actions to improve future decision-making and problem-solving.

3. **Entity Understanding**: By maintaining entity memory, agents can recognize and remember key entities, enhancing their ability to process and interact with complex information.

## Implementing Memory in Your Crew

When configuring a crew, you can enable and customize each memory component to suit the crew's objectives and the nature of tasks it will perform.
By default, the memory system is disabled, and you can ensure it is active by setting `memory=True` in the crew configuration. The memory will use OpenAI Embeddings by default, but you can change it by setting `embedder` to a different model.

The 'embedder' only applies to **Short-Term Memory** which uses Chroma for RAG using EmbedChain package.  
The **Long-Term Memory** uses SQLLite3 to store task results.  Currently, there is no way to override these storage implementations.
The data storage files are saved into a platform specific location found using the appdirs package 
and the name of the project which can be overridden using the **CREWAI_STORAGE_DIR** environment variable.

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

### Resetting Memory
```sh
crewai reset_memories [OPTIONS]
```

#### Resetting Memory Options
- **`-l, --long`**
  - **Description:** Reset LONG TERM memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-s, --short`**
  - **Description:** Reset SHORT TERM memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-e, --entities`**
  - **Description:** Reset ENTITIES memory.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-k, --kickoff-outputs`**
  - **Description:** Reset LATEST KICKOFF TASK OUTPUTS.
  - **Type:** Flag (boolean)
  - **Default:** False

- **`-a, --all`**
  - **Description:** Reset ALL memories.
  - **Type:** Flag (boolean)
  - **Default:** False



## Benefits of Using crewAI's Memory System
- **Adaptive Learning:** Crews become more efficient over time, adapting to new information and refining their approach to tasks.
- **Enhanced Personalization:** Memory enables agents to remember user preferences and historical interactions, leading to personalized experiences.
- **Improved Problem Solving:** Access to a rich memory store aids agents in making more informed decisions, drawing on past learnings and contextual insights.

## Getting Started
Integrating crewAI's memory system into your projects is straightforward. By leveraging the provided memory components and configurations, you can quickly empower your agents with the ability to remember, reason, and learn from their interactions, unlocking new levels of intelligence and capability.

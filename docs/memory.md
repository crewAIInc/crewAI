# Memory in CrewAI

CrewAI provides a robust memory system that allows agents to retain and recall information from previous interactions.

## Configuring Embedding Providers

CrewAI supports multiple embedding providers for memory functionality:

- OpenAI (default) - Requires `OPENAI_API_KEY`
- Ollama - Requires `CREWAI_OLLAMA_URL` (defaults to "http://localhost:11434/api/embeddings")

### Environment Variables

Configure the embedding provider using these environment variables:

- `CREWAI_EMBEDDING_PROVIDER`: Provider name (default: "openai")
- `CREWAI_EMBEDDING_MODEL`: Model name (default: "text-embedding-3-small")
- `CREWAI_OLLAMA_URL`: URL for Ollama API (when using Ollama provider)

### Example Configuration

```python
# Using OpenAI (default)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Using Ollama
os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
os.environ["CREWAI_EMBEDDING_MODEL"] = "llama2"  # or any other model supported by your Ollama instance
os.environ["CREWAI_OLLAMA_URL"] = "http://localhost:11434/api/embeddings"  # optional, this is the default
```

## Memory Usage

When an agent has memory enabled, it can access and store information from previous interactions:

```python
agent = Agent(
    role="Researcher",
    goal="Research AI topics",
    backstory="You're an AI researcher",
    memory=True  # Enable memory for this agent
)
```

The memory system uses embeddings to store and retrieve relevant information, allowing agents to maintain context across multiple interactions and tasks.

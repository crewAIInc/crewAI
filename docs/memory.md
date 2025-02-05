# Memory Management in CrewAI

CrewAI provides a robust memory system that allows agents to store and retrieve information across conversations and tasks.

## Memory Types

- **Short Term Memory**: Stores recent interactions and context
- **Long Term Memory**: Persists important information for extended periods
- **Entity Memory**: Tracks information about specific entities

## Configuration

### Embedding Providers

CrewAI supports multiple embedding providers for memory storage. By default, it uses OpenAI, but you can configure different providers:

```bash
# OpenAI (default)
export CREWAI_EMBEDDING_PROVIDER=openai
export CREWAI_EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_API_KEY=your_key

# Ollama
export CREWAI_EMBEDDING_PROVIDER=ollama
export CREWAI_EMBEDDING_MODEL=llama2
export CREWAI_OLLAMA_URL=http://localhost:11434/api/embeddings  # Optional
```

### Memory Operations

Reset all memories:
```bash
crewai reset-memories -a
```

The memory system will use the configured embedding provider for all operations, including memory reset.

## Troubleshooting

### Common Issues

1. OpenAI API Key Missing
```bash
# Error: EmbeddingConfigurationError: Invalid configuration for OpenAI provider
# Solution: Set your OpenAI API key
export OPENAI_API_KEY=your_key
```

2. Ollama Connection Issues
```bash
# Error: Failed to connect to Ollama server
# Solution: Ensure Ollama is running and accessible
curl http://localhost:11434/api/embeddings
```

3. Memory Reset Permission Issues
```bash
# Error: Failed to reset memory (readonly database)
# Solution: Check file permissions or use custom path
export CREWAI_MEMORY_PATH=/path/with/write/access
```

### Custom Storage Path

You can configure a custom storage path for memory files:

```python
from crewai.memory import ShortTermMemory

memory = ShortTermMemory(path="/custom/storage/path")
```

This is useful when:
- Default storage location has permission issues
- You want to isolate memory storage for different crews
- You need to manage memory persistence manually

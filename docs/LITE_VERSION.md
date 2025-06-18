# CrewAI Lite Version

CrewAI now supports a "lite" installation with minimal dependencies, allowing you to use the core functionality without installing heavy optional dependencies.

## Installation

### Lite Installation (Minimal Dependencies)
```bash
pip install crewai
```

This installs only the core dependencies needed for basic Agent, Crew, and Task functionality.

### Full Installation (All Dependencies)
```bash
pip install crewai[all]
```

### Selective Installation (Optional Extras)

Install only the features you need:

```bash
# Memory and knowledge storage
pip install crewai[memory]

# Knowledge sources (PDF, Excel, etc.)
pip install crewai[knowledge]

# Telemetry and monitoring
pip install crewai[telemetry]

# Flow visualization
pip install crewai[visualization]

# Authentication features
pip install crewai[auth]

# Additional LLM integrations
pip install crewai[llm-integrations]

# AgentOps integration
pip install crewai[agentops]

# FastEmbed embeddings
pip install crewai[embeddings]
```

You can also combine multiple extras:
```bash
pip install crewai[memory,knowledge,telemetry]
```

## Core vs Optional Features

### Core Features (Always Available)
- Basic Agent, Crew, and Task functionality
- LiteAgent for simple interactions
- Core LLM integrations (OpenAI, etc.)
- Basic tools and utilities
- Process management

### Optional Features (Require Extras)

#### Memory (`crewai[memory]`)
- RAG storage with ChromaDB
- Memory management
- Embeddings configuration

#### Knowledge (`crewai[knowledge]`)
- PDF knowledge sources
- Excel/spreadsheet processing
- Document processing with Docling
- Knowledge storage and retrieval

#### Telemetry (`crewai[telemetry]`)
- OpenTelemetry integration
- Performance monitoring
- Usage analytics

#### Visualization (`crewai[visualization]`)
- Flow visualization with Pyvis
- Network diagrams

#### Authentication (`crewai[auth]`)
- Auth0 integration
- Secure token management

#### LLM Integrations (`crewai[llm-integrations]`)
- AISuite integration
- Additional model providers

## Error Handling

When you try to use a feature that requires optional dependencies, you'll get a helpful error message:

```python
from crewai.memory.storage.rag_storage import RAGStorage

# Without crewai[memory] installed:
# ImportError: ChromaDB is required for RAG storage functionality. 
# Please install it with: pip install 'crewai[memory]'
```

## Migration Guide

Existing installations will continue to work as before. If you want to switch to the lite version:

1. Uninstall current crewai: `pip uninstall crewai`
2. Install lite version: `pip install crewai`
3. Add extras as needed: `pip install crewai[memory,knowledge]`

## Benefits

- **Reduced installation size**: Core installation is much smaller
- **Faster installation**: Fewer dependencies to download and compile
- **Reduced security surface**: Fewer dependencies means fewer potential vulnerabilities
- **Flexible**: Install only what you need
- **Backward compatible**: Existing code continues to work with full installation

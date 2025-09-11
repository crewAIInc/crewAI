---
description: Repository Information Overview
alwaysApply: true
---

# CrewAI Information

## Summary
CrewAI is a Python framework for orchestrating role-playing, autonomous AI agents. It enables developers to create teams of specialized AI agents that work together to accomplish complex tasks through collaborative intelligence. The framework is built from scratch, independent of LangChain or other agent frameworks, focusing on performance and flexibility.

## Structure
- **src/crewai**: Core framework code with agents, tasks, crews, and flows
- **tests**: Comprehensive test suite covering all framework components
- **docs**: Documentation resources and images
- **examples**: (Not visible in current view but referenced in README)

## Language & Runtime
**Language**: Python
**Version**: >=3.10, <3.14
**Build System**: Hatch (hatchling)
**Package Manager**: pip/uv

## Dependencies
**Main Dependencies**:
- pydantic (>=2.4.2): Data validation and settings management
- openai (>=1.13.3): OpenAI API integration
- litellm (==1.74.9): LLM abstraction layer
- instructor (>=1.3.3): Structured outputs
- chromadb (>=0.5.23): Vector database for embeddings
- tokenizers (>=0.20.3): Text tokenization

**Development Dependencies**:
- ruff (>=0.12.11): Python linter
- mypy (>=1.17.1): Static type checking
- pytest (>=8.0.0): Testing framework
- pre-commit (>=4.3.0): Git hooks manager

## Build & Installation
```bash
pip install crewai
# With additional tools
pip install 'crewai[tools]'
# With embeddings support
pip install 'crewai[embeddings]'
```

## Testing
**Framework**: pytest
**Test Location**: /tests directory with subdirectories for components
**Naming Convention**: test_*.py files
**Configuration**: pytest.ini_options in pyproject.toml
**Run Command**:
```bash
pytest
```

## CLI Tools
**Command**: crewai
**Features**:
- Create new crew projects: `crewai create crew <project_name>`
- Run crews: `crewai run <project_name>`
- Train crews: `crewai train <project_name>`
- Evaluate crews: `crewai evaluate <project_name>`

## Main Components
- **Agent**: Autonomous AI agents with specific roles, goals, and tools
- **Task**: Units of work assigned to agents with descriptions and expected outputs
- **Crew**: Teams of agents working together through sequential or hierarchical processes
- **Flow**: Event-driven workflows for precise control over complex automations
- **Knowledge**: Integration with knowledge sources for agent context
- **Memory**: Short-term and long-term memory systems for agents
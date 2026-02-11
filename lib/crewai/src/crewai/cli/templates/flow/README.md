# {{crew_name}} Flow

Welcome to the {{crew_name}} Flow project, powered by [crewAI](https://crewai.com). This template demonstrates how to build a multi-agent AI system using CrewAI's Flow-based architecture with state management. The project showcases enterprise best practices for orchestrating AI agents, managing state with Pydantic models, and organizing crews using YAML-based configuration.

This template includes a working **Poem Generation Flow** that demonstrates:

- **Flow-based Architecture**: Sequential execution with `@start()` and `@listen()` decorators
- **State Management**: Type-safe state passing using Pydantic models
- **YAML Configuration**: Agents and tasks defined in YAML files, not Python code
- **Crew Integration**: PoemCrew with a single agent that generates creative poems about CrewAI

## Project Structure

```
template_flow/
├── src/template_flow/
│   ├── main.py                      # Flow orchestration and entry points
│   ├── crews/
│   │   └── poem_crew/              # Poetry generation crew
│   │       ├── poem_crew.py        # Crew class with @CrewBase decorator
│   │       └── config/
│   │           ├── agents.yaml     # Agent definitions
│   │           └── tasks.yaml      # Task definitions
│   └── tools/
│       └── custom_tool.py          # Example custom tool template
├── pyproject.toml                  # Project metadata and dependencies
├── .env                            # Environment configuration (add your API key here)
└── README.md
```

## Installation

Ensure you have Python installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
uv sync
```

### Environment Setup

**Add your `OPENAI_API_KEY` to the `.env` file:**

```bash
OPENAI_API_KEY=your_api_key_here
```

## Running the Project

To kickstart your flow and begin execution, run this from the root folder of your project:

```bash
crewai run
```

This will:
1. Generate a random sentence count (1-5)
2. Create a poem about CrewAI with that many sentences
3. Save the poem to `poem.txt` in the root directory

### Visualization

To visualize the flow structure:

```bash
crewai plot
```

This generates a visual representation of the flow's execution graph.

## Understanding the Flow Architecture

### Flow State Management

The `PoemFlow` class uses Pydantic models for type-safe state management:

```python
class PoemFlowState(BaseModel):
    sentence_count: int = 1
    poem: str = ""
```

State is shared across all flow steps and can be accessed via `self.state`.

### Flow Execution Steps

1. **`@start() generate_sentence_count()`**
   - Entry point that initializes the sentence count
   - Can accept trigger payloads for dynamic parameters
   - Updates `self.state.sentence_count`

2. **`@listen(generate_sentence_count) generate_poem()`**
   - Executes the PoemCrew to generate a poem
   - Uses the sentence count from state
   - Updates `self.state.poem` with the result

3. **`@listen(generate_poem) save_poem()`**
   - Saves the generated poem to `poem.txt`
   - Final step in the flow

### PoemCrew Details

**Agent: poem_writer**
- Role: CrewAI Poem Writer
- Goal: Generate funny, light-hearted poems about CrewAI
- Configuration: Defined in `src/template_flow/crews/poem_crew/config/agents.yaml`

**Task: write_poem**
- Description: Write an engaging poem about CrewAI
- Expected Output: A beautifully crafted poem with the exact sentence count
- Configuration: Defined in `src/template_flow/crews/poem_crew/config/tasks.yaml`

## Customizing the Template

### Adding New Crews

1. Create a new directory under `src/template_flow/crews/`
2. Add `config/agents.yaml` and `config/tasks.yaml`
3. Create a crew class with `@CrewBase` decorator following the pattern in `poem_crew.py`
4. Import and use in your flow with `crew.crew().kickoff()`

### Creating Custom Tools

Use the template in `src/template_flow/tools/custom_tool.py`:

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class MyCustomToolInput(BaseModel):
    argument: str = Field(..., description="Description")

class MyCustomTool(BaseTool):
    name: str = "Tool Name"
    description: str = "What this tool does"
    args_schema: type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Your implementation
        return "result"
```

### Modifying the Flow

Edit `src/template_flow/main.py` to:
- Add new flow steps with `@listen()` decorator
- Implement conditional routing with `@router()` decorator
- Add parallel execution with `@or_()` or `@and_()` functions
- Expand the state model with additional fields

### Updating Agent/Task Configuration

- Edit `agents.yaml` to modify agent roles, goals, and backstories
- Edit `tasks.yaml` to change task descriptions and expected outputs
- Use variable interpolation with `{variable_name}` syntax for dynamic values

## Best Practices

This template follows enterprise best practices for CrewAI development:

- **YAML-First Configuration**: Agents and tasks defined in YAML, not Python classes
- **Flow-Based Architecture**: Uses `Flow` class with state management instead of standalone crews
- **Minimal Crew Classes**: Crews are just wiring, configuration lives in YAML
- **Type Safety**: Pydantic models for state and tool inputs
- **Clean Code**: Follows PEP 8, single responsibility principle, and DRY principles

## Support

For support, questions, or feedback regarding CrewAI:

- Visit the [CrewAI documentation](https://docs.crewai.com)
- Check the [GitHub repository](https://github.com/crewAIInc/crewai)
- [Chat with the docs](https://chatg.pt/DWjSBZn)

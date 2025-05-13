# CrewAI CLI Documentation

The CrewAI Command Line Interface (CLI) provides tools for creating, managing, and running CrewAI projects.

## Installation

The CLI is automatically installed when you install the CrewAI package:

```bash
pip install crewai
```

## Available Commands

### Create Command

The `create` command allows you to create new crews or flows.

```bash
crewai create [TYPE] [NAME] [OPTIONS]
```

#### Arguments

- `TYPE`: Type of project to create. Must be either `crew` or `flow`.
- `NAME`: Name of the project to create.

#### Options

- `--provider`: The provider to use for the crew.
- `--skip_provider`: Skip provider validation.
- `--skip_ssl_verify`: Skip SSL certificate verification when fetching provider data (not secure).

#### Examples

Create a new crew:

```bash
crewai create crew my_crew
```

Create a new crew with a specific provider:

```bash
crewai create crew my_crew --provider openai
```

Create a new crew and skip SSL certificate verification (useful in environments with self-signed certificates):

```bash
crewai create crew my_crew --skip_ssl_verify
```

> **Warning**: Using the `--skip_ssl_verify` flag is not recommended in production environments as it bypasses SSL certificate verification, which can expose your system to security risks. Only use this flag in development environments or when you understand the security implications.

Create a new flow:

```bash
crewai create flow my_flow
```

### Run Command

The `run` command executes your crew.

```bash
crewai run
```

### Train Command

The `train` command trains your crew.

```bash
crewai train [OPTIONS]
```

#### Options

- `-n, --n_iterations`: Number of iterations to train the crew (default: 5).
- `-f, --filename`: Path to a custom file for training (default: "trained_agents_data.pkl").

### Reset Memories Command

The `reset_memories` command resets the crew memories.

```bash
crewai reset_memories [OPTIONS]
```

#### Options

- `-l, --long`: Reset LONG TERM memory.
- `-s, --short`: Reset SHORT TERM memory.
- `-e, --entities`: Reset ENTITIES memory.
- `-kn, --knowledge`: Reset KNOWLEDGE storage.
- `-k, --kickoff-outputs`: Reset LATEST KICKOFF TASK OUTPUTS.
- `-a, --all`: Reset ALL memories.

### Other Commands

- `version`: Show the installed version of crewai.
- `replay`: Replay the crew execution from a specific task.
- `log_tasks_outputs`: Retrieve your latest crew.kickoff() task outputs.
- `test`: Test the crew and evaluate the results.
- `install`: Install the Crew.
- `update`: Update the pyproject.toml of the Crew project to use uv.
- `signup`: Sign Up/Login to CrewAI+.
- `login`: Sign Up/Login to CrewAI+.
- `chat`: Start a conversation with the Crew.

## Security Considerations

When using the CrewAI CLI, be aware of the following security considerations:

1. **API Keys**: Store your API keys securely in environment variables or a `.env` file. Never hardcode them in your scripts.

2. **SSL Verification**: The `--skip_ssl_verify` flag bypasses SSL certificate verification, which can expose your system to security risks. Only use this flag in development environments or when you understand the security implications.

3. **Provider Data**: When fetching provider data, ensure that you're using a secure connection. The CLI will display a warning when SSL verification is disabled.

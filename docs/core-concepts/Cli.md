# CrewAI CLI Documentation

The CrewAI CLI provides a set of commands to interact with CrewAI, allowing you to create, train, run, and manage crews and pipelines.

## Installation

To use the CrewAI CLI, make sure you have CrewAI & Poetry installed:

```
pip install crewai poetry
```

## Basic Usage

The basic structure of a CrewAI CLI command is:

```
crewai [COMMAND] [OPTIONS] [ARGUMENTS]
```

## Available Commands

### 1. create

Create a new crew or pipeline.

```
crewai create [OPTIONS] TYPE NAME
```

- `TYPE`: Choose between "crew" or "pipeline"
- `NAME`: Name of the crew or pipeline
- `--router`: (Optional) Create a pipeline with router functionality

Example:
```
crewai create crew my_new_crew
crewai create pipeline my_new_pipeline --router
```

### 2. version

Show the installed version of CrewAI.

```
crewai version [OPTIONS]
```

- `--tools`: (Optional) Show the installed version of CrewAI tools

Example:
```
crewai version
crewai version --tools
```

### 3. train

Train the crew for a specified number of iterations.

```
crewai train [OPTIONS]
```

- `-n, --n_iterations INTEGER`: Number of iterations to train the crew (default: 5)
- `-f, --filename TEXT`: Path to a custom file for training (default: "trained_agents_data.pkl")

Example:
```
crewai train -n 10 -f my_training_data.pkl
```

### 4. replay

Replay the crew execution from a specific task.

```
crewai replay [OPTIONS]
```

- `-t, --task_id TEXT`: Replay the crew from this task ID, including all subsequent tasks

Example:
```
crewai replay -t task_123456
```

### 5. log_tasks_outputs

Retrieve your latest crew.kickoff() task outputs.

```
crewai log_tasks_outputs
```

### 6. reset_memories

Reset the crew memories (long, short, entity, latest_crew_kickoff_outputs).

```
crewai reset_memories [OPTIONS]
```

- `-l, --long`: Reset LONG TERM memory
- `-s, --short`: Reset SHORT TERM memory
- `-e, --entities`: Reset ENTITIES memory
- `-k, --kickoff-outputs`: Reset LATEST KICKOFF TASK OUTPUTS
- `-a, --all`: Reset ALL memories

Example:
```
crewai reset_memories --long --short
crewai reset_memories --all
```

### 7. test

Test the crew and evaluate the results.

```
crewai test [OPTIONS]
```

- `-n, --n_iterations INTEGER`: Number of iterations to test the crew (default: 3)
- `-m, --model TEXT`: LLM Model to run the tests on the Crew (default: "gpt-4o-mini")

Example:
```
crewai test -n 5 -m gpt-3.5-turbo
```

### 8. run

Run the crew.

```
crewai run
```

## Note

Make sure to run these commands from the directory where your CrewAI project is set up. Some commands may require additional configuration or setup within your project structure.

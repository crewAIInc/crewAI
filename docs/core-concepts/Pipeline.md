---
title: crewAI Pipelines
description: Understanding and utilizing pipelines in the crewAI framework for efficient multi-stage task processing.
---

## What is a Pipeline?

A pipeline in crewAI represents a structured workflow that allows for the sequential or parallel execution of multiple crews. It provides a way to organize complex processes involving multiple stages, where the output of one stage can serve as input for subsequent stages.

## Key Terminology

Understanding the following terms is crucial for working effectively with pipelines:

- **Stage**: A distinct part of the pipeline, which can be either sequential (a single crew) or parallel (multiple crews executing concurrently).
- **Run**: A specific execution of the pipeline for a given set of inputs, representing a single instance of processing through the pipeline.
- **Branch**: Parallel executions within a stage (e.g., concurrent crew operations).
- **Trace**: The journey of an individual input through the entire pipeline, capturing the path and transformations it undergoes.

Example pipeline structure:

```
crew1 >> [crew2, crew3] >> crew4
```

This represents a pipeline with three stages:

1. A sequential stage (crew1)
2. A parallel stage with two branches (crew2 and crew3 executing concurrently)
3. Another sequential stage (crew4)

Each input creates its own run, flowing through all stages of the pipeline. Multiple runs can be processed concurrently, each following the defined pipeline structure.

## Pipeline Attributes

| Attribute  | Parameters | Description                                                                           |
| :--------- | :--------- | :------------------------------------------------------------------------------------ |
| **Stages** | `stages`   | A list of crews or lists of crews representing the stages to be executed in sequence. |

## Creating a Pipeline

When creating a pipeline, you define a series of stages, each consisting of either a single crew or a list of crews for parallel execution. The pipeline ensures that each stage is executed in order, with the output of one stage feeding into the next.

### Example: Assembling a Pipeline

```python
from crewai import Crew, Agent, Task, Pipeline

# Define your crews
research_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

analysis_crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    process=Process.sequential
)

writing_crew = Crew(
    agents=[writer],
    tasks=[writing_task],
    process=Process.sequential
)

# Assemble the pipeline
my_pipeline = Pipeline(
    stages=[research_crew, analysis_crew, writing_crew]
)
```

## Pipeline Methods

| Method           | Description                                                                                                                                                                    |
| :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **process_runs** | Executes the pipeline, processing all stages and returning the results. This method initiates one or more runs through the pipeline, handling the flow of data between stages. |

## Pipeline Output

!!! note "Understanding Pipeline Outputs"
The output of a pipeline in the crewAI framework is encapsulated within two main classes: `PipelineOutput` and `PipelineRunResult`. These classes provide a structured way to access the results of the pipeline's execution, including various formats such as raw strings, JSON, and Pydantic models.

### Pipeline Output Attributes

| Attribute       | Parameters    | Type                      | Description                                                                                               |
| :-------------- | :------------ | :------------------------ | :-------------------------------------------------------------------------------------------------------- |
| **ID**          | `id`          | `UUID4`                   | A unique identifier for the pipeline output.                                                              |
| **Run Results** | `run_results` | `List[PipelineRunResult]` | A list of `PipelineRunResult` objects, each representing the output of a single run through the pipeline. |

### Pipeline Output Methods

| Method/Property    | Description                                            |
| :----------------- | :----------------------------------------------------- |
| **add_run_result** | Adds a `PipelineRunResult` to the list of run results. |

### Pipeline Run Result Attributes

| Attribute         | Parameters      | Type                       | Description                                                                                   |
| :---------------- | :-------------- | :------------------------- | :-------------------------------------------------------------------------------------------- |
| **ID**            | `id`            | `UUID4`                    | A unique identifier for the run result.                                                       |
| **Raw**           | `raw`           | `str`                      | The raw output of the final stage in the pipeline run.                                        |
| **Pydantic**      | `pydantic`      | `Optional[BaseModel]`      | A Pydantic model object representing the structured output of the final stage, if applicable. |
| **JSON Dict**     | `json_dict`     | `Optional[Dict[str, Any]]` | A dictionary representing the JSON output of the final stage, if applicable.                  |
| **Token Usage**   | `token_usage`   | `Dict[str, Any]`           | A summary of token usage across all stages of the pipeline run.                               |
| **Trace**         | `trace`         | `List[Any]`                | A trace of the journey of inputs through the pipeline run.                                    |
| **Crews Outputs** | `crews_outputs` | `List[CrewOutput]`         | A list of `CrewOutput` objects, representing the outputs from each crew in the pipeline run.  |

### Pipeline Run Result Methods and Properties

| Method/Property | Description                                                                                              |
| :-------------- | :------------------------------------------------------------------------------------------------------- |
| **json**        | Returns the JSON string representation of the run result if the output format of the final task is JSON. |
| **to_dict**     | Converts the JSON and Pydantic outputs to a dictionary.                                                  |
| \***\*str\*\*** | Returns the string representation of the run result, prioritizing Pydantic, then JSON, then raw.         |

### Accessing Pipeline Outputs

Once a pipeline has been executed, its output can be accessed through the `PipelineOutput` object returned by the `process_runs` method. The `PipelineOutput` class provides access to individual `PipelineRunResult` objects, each representing a single run through the pipeline.

#### Example

```python
# Define input data for the pipeline
input_data = [{"initial_query": "Latest advancements in AI"}, {"initial_query": "Future of robotics"}]

# Execute the pipeline
pipeline_output = await my_pipeline.process_runs(input_data)

# Access the results
for run_result in pipeline_output.run_results:
    print(f"Run ID: {run_result.id}")
    print(f"Final Raw Output: {run_result.raw}")
    if run_result.json_dict:
        print(f"JSON Output: {json.dumps(run_result.json_dict, indent=2)}")
    if run_result.pydantic:
        print(f"Pydantic Output: {run_result.pydantic}")
    print(f"Token Usage: {run_result.token_usage}")
    print(f"Trace: {run_result.trace}")
    print("Crew Outputs:")
    for crew_output in run_result.crews_outputs:
        print(f"  Crew: {crew_output.raw}")
    print("\n")
```

This example demonstrates how to access and work with the pipeline output, including individual run results and their associated data.

## Using Pipelines

Pipelines are particularly useful for complex workflows that involve multiple stages of processing, analysis, or content generation. They allow you to:

1. **Sequence Operations**: Execute crews in a specific order, ensuring that the output of one crew is available as input to the next.
2. **Parallel Processing**: Run multiple crews concurrently within a stage for increased efficiency.
3. **Manage Complex Workflows**: Break down large tasks into smaller, manageable steps executed by specialized crews.

### Example: Running a Pipeline

```python
# Define input data for the pipeline
input_data = [{"initial_query": "Latest advancements in AI"}]

# Execute the pipeline, initiating a run for each input
results = await my_pipeline.process_runs(input_data)

# Access the results
for result in results:
    print(f"Final Output: {result.raw}")
    print(f"Token Usage: {result.token_usage}")
    print(f"Trace: {result.trace}")  # Shows the path of the input through all stages
```

## Advanced Features

### Parallel Execution within Stages

You can define parallel execution within a stage by providing a list of crews, creating multiple branches:

```python
parallel_analysis_crew = Crew(agents=[financial_analyst], tasks=[financial_analysis_task])
market_analysis_crew = Crew(agents=[market_analyst], tasks=[market_analysis_task])

my_pipeline = Pipeline(
    stages=[
        research_crew,
        [parallel_analysis_crew, market_analysis_crew],  # Parallel execution (branching)
        writing_crew
    ]
)
```

### Error Handling and Validation

The Pipeline class includes validation mechanisms to ensure the robustness of the pipeline structure:

- Validates that stages contain only Crew instances or lists of Crew instances.
- Prevents double nesting of stages to maintain a clear structure.

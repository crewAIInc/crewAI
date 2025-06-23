# Flow-Level RPM Control

CrewAI now supports global RPM (Requests Per Minute) control at the Flow level, allowing you to set a single rate limit that applies across all Crews within a Flow. This feature is essential for complex applications that orchestrate multiple Crews and need precise control over API usage rates.

## Overview

Previously, CrewAI provided RPM control at the Agent and Crew levels independently. When working with Flows that coordinate multiple Crews, there was no mechanism to control the overall request rate across all Crews within a Flow. This could lead to exceeding API rate limits when multiple Crews executed simultaneously.

With Flow-level RPM control, you can now:

- Set a global RPM limit for all Crews in a Flow
- Prevent API rate limit errors in complex multi-Crew workflows
- Simplify rate management across multiple Crews
- Better control API costs in production applications

## Key Features

- **Global Rate Limiting**: One RPM limit applies to all Crews within the Flow
- **Automatic Configuration**: Crews created within Flow methods are automatically configured
- **Thread-Safe**: Handles concurrent Crew execution safely
- **Override Behavior**: Flow-level limits override individual Crew and Agent RPM settings
- **Backward Compatible**: Existing Flows without RPM limits work unchanged

## Usage

### Basic Flow with RPM Control

```python
from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start, listen

class AnalysisFlow(Flow):
    def __init__(self):
        # Set global RPM limit of 10 requests per minute for entire Flow
        super().__init__(max_rpm=10, verbose=True)

    @start()
    def initialize_analysis(self):
        return {"status": "initialized"}

    @listen(initialize_analysis)
    def run_data_collection_crew(self, context):
        # Create agents
        data_analyst = Agent(
            role="Data Analyst",
            goal="Collect and validate data",
            backstory="Expert in data extraction",
            max_rpm=20  # This will be overridden by Flow's 10 RPM limit
        )

        # Create tasks
        collect_task = Task(
            description="Collect data from sources",
            agent=data_analyst,
            expected_output="Clean dataset"
        )

        # Create crew - automatically uses Flow's RPM limit
        data_crew = Crew(
            agents=[data_analyst],
            tasks=[collect_task],
            max_rpm=15  # This will be overridden by Flow's 10 RPM limit
        )

        return data_crew.kickoff()

    @listen(run_data_collection_crew)
    def run_analysis_crew(self, context):
        # This crew will also be limited to Flow's 10 RPM
        analysis_agent = Agent(
            role="Data Analyst",
            goal="Analyze collected data",
            backstory="Expert in data analysis"
        )

        analysis_task = Task(
            description="Analyze the data",
            agent=analysis_agent,
            expected_output="Analysis report"
        )

        analysis_crew = Crew(
            agents=[analysis_agent],
            tasks=[analysis_task]
        )

        return analysis_crew.kickoff()

# Execute the flow
flow = AnalysisFlow()
result = flow.kickoff()
```

### Flow Without RPM Control

```python
class UnlimitedFlow(Flow):
    @start()
    def process_data(self):
        agent = Agent(
            role="Data Processor",
            goal="Process data quickly",
            backstory="High-performance specialist"
        )

        task = Task(
            description="Process dataset",
            agent=agent,
            expected_output="Processed data"
        )

        # This crew uses its own RPM settings
        crew = Crew(
            agents=[agent],
            tasks=[task],
            max_rpm=50  # This will be respected
        )

        return crew.kickoff()

# No global RPM limit
flow = UnlimitedFlow()
result = flow.kickoff()
```

### Manual Crew Configuration

```python
# Create flow with RPM control
flow = AnalysisFlow()

# Create crew manually outside of flow methods
agent = Agent(role="Manual Agent", goal="Test", backstory="Test agent")
task = Task(description="Test task", agent=agent, expected_output="Result")
crew = Crew(agents=[agent], tasks=[task], max_rpm=25)

# Manually apply flow's RPM controller
flow.set_crew_rpm_controller(crew)

# Now crew uses flow's RPM limit
crew.kickoff()
```

## API Reference

### Flow Class

#### Constructor Parameters

```python
Flow(
    persistence: Optional[FlowPersistence] = None,
    max_rpm: Optional[int] = None,
    verbose: bool = False,
    **kwargs: Any
)
```

- **`max_rpm`**: Maximum requests per minute for all Crews in this Flow
- **`verbose`**: Enable verbose logging for RPM operations
- **`persistence`**: Optional persistence backend
- **`**kwargs`**: Additional state initialization values

#### Methods

##### `get_flow_rpm_controller() -> Optional[RPMController]`

Returns the Flow's global RPM controller, or `None` if no RPM limit is set.

```python
flow = AnalysisFlow()
controller = flow.get_flow_rpm_controller()
if controller:
    print(f"Flow RPM limit: {controller.max_rpm}")
```

##### `set_crew_rpm_controller(crew: Crew) -> None`

Manually configure a Crew to use the Flow's global RPM controller.

```python
flow.set_crew_rpm_controller(my_crew)
```

### Crew Class

#### New Method

##### `set_flow_rpm_controller(rpm_controller: RPMController) -> None`

Set an external RPM controller (typically from a Flow) on this Crew.

```python
crew = Crew(agents=[agent], tasks=[task])
crew.set_flow_rpm_controller(flow_controller)
```

## How It Works

### Automatic Configuration

When a Flow method returns a Crew instance (or data structure containing Crews), the Flow automatically configures those Crews to use its global RPM controller:

1. **Method Execution**: Flow executes a method decorated with `@start()` or `@listen()`
2. **Result Processing**: Flow inspects the returned result for Crew instances
3. **Auto-Configuration**: Any found Crews are configured with the Flow's RPM controller
4. **Agent Updates**: All agents within configured Crews are also updated

### Override Behavior

When a Flow has a global RPM limit:

- **Crew RPM Settings**: Individual Crew `max_rpm` settings are overridden
- **Agent RPM Settings**: Individual Agent `max_rpm` settings are overridden
- **Shared Rate Limiting**: All Crews and Agents share the same RPM controller instance

### Thread Safety

The RPM controller uses thread-safe mechanisms:

- **Locking**: Thread locks prevent race conditions during rate limit checks
- **Atomic Operations**: Request counting and timing operations are atomic
- **Concurrent Crews**: Multiple Crews can execute concurrently while respecting the global limit

## Best Practices

### 1. Choose Appropriate RPM Limits

```python
# For development/testing
class DevFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=5)  # Conservative limit

# For production with higher requirements
class ProdFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=60)  # Higher limit for production
```

### 2. Monitor RPM Usage

```python
class MonitoredFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=20, verbose=True)  # Enable logging

    @start()
    def monitored_process(self):
        controller = self.get_flow_rpm_controller()
        if controller:
            print(f"Current RPM limit: {controller.max_rpm}")
        # ... rest of method
```

### 3. Handle Rate Limiting Gracefully

```python
@listen(some_method)
def rate_limited_process(self, context):
    try:
        # Create and execute crew
        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()
        return result
    except Exception as e:
        if "RPM" in str(e):
            print("Rate limit reached, will retry after delay")
            # Handle rate limiting appropriately
        raise
```

### 4. Design for Scalability

```python
class ScalableFlow(Flow):
    def __init__(self, environment="dev"):
        # Adjust RPM based on environment
        rpm_limits = {
            "dev": 10,
            "staging": 30,
            "prod": 100
        }
        super().__init__(max_rpm=rpm_limits.get(environment, 10))
```

## Migration Guide

### From Individual Crew RPM to Flow RPM

**Before:**
```python
# Manual coordination of RPM across crews
crew1 = Crew(agents=[agent1], tasks=[task1], max_rpm=5)
crew2 = Crew(agents=[agent2], tasks=[task2], max_rpm=5)
# Total: 10 RPM, but no coordination between crews
```

**After:**
```python
class CoordinatedFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=10)  # Global limit for both crews

    @start()
    def run_crew1(self):
        crew1 = Crew(agents=[agent1], tasks=[task1])  # Uses global limit
        return crew1.kickoff()

    @listen(run_crew1)
    def run_crew2(self, context):
        crew2 = Crew(agents=[agent2], tasks=[task2])  # Uses global limit
        return crew2.kickoff()
```

## Troubleshooting

### Common Issues

#### 1. Crews Not Using Flow RPM Limit

**Problem**: Manually created Crews aren't using the Flow's RPM limit.

**Solution**: Use `set_crew_rpm_controller()` for manual configuration:

```python
flow = MyFlow()
crew = Crew(agents=[agent], tasks=[task])
flow.set_crew_rpm_controller(crew)  # Apply Flow's RPM limit
```

#### 2. Rate Limits Still Being Exceeded

**Problem**: API rate limits are still being hit despite setting Flow RPM.

**Solution**: Check that the Flow RPM limit is appropriate for your API provider:

```python
# Check your API provider's rate limits
# OpenAI: typically 3-60 RPM depending on tier
# Anthropic: varies by plan
# Adjust Flow RPM accordingly
class ConservativeFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=3)  # Very conservative
```

#### 3. Flows Running Slower Than Expected

**Problem**: Flow execution is slower after adding RPM control.

**Solution**: The RPM limit may be too restrictive. Monitor and adjust:

```python
class OptimizedFlow(Flow):
    def __init__(self):
        # Start conservative, then increase based on monitoring
        super().__init__(max_rpm=30, verbose=True)
```

### Debug Mode

Enable verbose logging to debug RPM issues:

```python
class DebugFlow(Flow):
    def __init__(self):
        super().__init__(max_rpm=10, verbose=True)

    @start()
    def debug_method(self):
        print(f"Flow RPM controller: {self.get_flow_rpm_controller()}")
        # Create crew and check its controller
        crew = Crew(agents=[agent], tasks=[task])
        print(f"Crew RPM controller: {crew._rpm_controller}")
        return crew.kickoff()
```

## Examples Repository

Find more examples in the `examples/` directory:

- `flow_rpm_control_example.py`: Complete working example
- `flow_rpm_migration_example.py`: Migration from individual to Flow RPM
- `flow_rpm_production_example.py`: Production-ready implementation

## Related Documentation

- [Flow Basics](./flows.md)
- [RPM Controller](./rpm_controller.md)
- [Crew Configuration](./crews.md)
- [Agent Configuration](./agents.md)

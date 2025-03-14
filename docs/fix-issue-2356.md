# Fix for Issue #2356: Missing Parentheses in Flow Documentation

## Issue
In the "first-flow.mdx" documentation, there's an error in the code example where a task method reference is missing parentheses:

```python
@task
def review_section_task(self) -> Task:
    return Task(
        config=self.tasks_config['review_section_task'],
        context=[self.write_section_task]  # Missing parentheses
    )
```

This causes an AttributeError when running `crewai flow kickoff` because the Flow system requires explicit method calls with parentheses.

## Error Message
When users follow the documentation and use the code as shown, they encounter this error:
```
AttributeError: 'function' object has no attribute 'get'
```

## Root Cause
The core issue is that the Flow system in CrewAI requires explicit method calls with parentheses when processing context tasks. This is implemented in the `_map_task_variables` method in `crew_base.py`:

```python
if context_list := task_info.get("context"):
    self.tasks_config[task_name]["context"] = [
        tasks[context_task_name]() for context_task_name in context_list
    ]
```

When users follow the documentation and use `context=[self.write_section_task]` without parentheses, they get an AttributeError because a function object doesn't have a `get` attribute.

## Fix
The correct code should be:

```python
@task
def review_section_task(self) -> Task:
    return Task(
        config=self.tasks_config['review_section_task'],
        context=[self.write_section_task()]  # Added parentheses
    )
```

## Verification
I've created a minimal reproducible example that demonstrates both the error and the fix. The error occurs because in `crew_base.py`, the `_map_task_variables` method explicitly requires method calls with parentheses when processing context tasks.

## Documentation Update Needed
The documentation at docs.crewai.com/guides/flows/first-flow needs to be updated to show the correct syntax with parentheses.

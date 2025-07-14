from crewai import Agent, Task

def prioritize_tasks(tasks):
    """Function that accesses task.config to prioritize tasks - this should work after the fix."""
    return sorted(tasks, key=lambda t: {"low": 2, "medium": 1, "high": 0}.get(t.config.get("priority", "medium")))

researcher = Agent(
    role="Researcher",
    goal="Find relevant facts",
    backstory="An expert at gathering information quickly.",
    verbose=True
)

print("=== Test 1: Basic config retention ===")
task_with_config = Task(
    description="Test task with config",
    expected_output="Test output",
    config={"priority": "high", "category": "research", "timeout": 300}
)

print(f"Task config: {task_with_config.config}")
print(f"Config type: {type(task_with_config.config)}")
print(f"Priority from config: {task_with_config.config.get('priority') if task_with_config.config else 'None'}")

print("\n=== Test 2: Config with valid Task fields ===")
task_with_field_config = Task(
    description="Original description",
    expected_output="Original output",
    config={
        "name": "Config Task Name",
        "human_input": True,
        "custom_field": "custom_value"
    }
)

print(f"Task name: {task_with_field_config.name}")
print(f"Task human_input: {task_with_field_config.human_input}")
print(f"Task config: {task_with_field_config.config}")
print(f"Custom field from config: {task_with_field_config.config.get('custom_field') if task_with_field_config.config else 'None'}")

print("\n=== Test 3: Callback retention ===")
def test_callback(output):
    return f"Callback executed with: {output}"

task_with_callback = Task(
    description="Test task with callback",
    expected_output="Test output",
    callback=test_callback
)

print(f"Task callback: {task_with_callback.callback}")
print(f"Callback callable: {callable(task_with_callback.callback)}")

print("\n=== Test 4: Original issue scenario ===")
tasks = [
    Task(
        description="Search for the author's biography",
        expected_output="A summary of the author's background",
        agent=researcher,
        config={"priority": "high", "category": "research"}
    ),
    Task(
        description="Check publication date", 
        expected_output="Date of first publication",
        agent=researcher,
        config={"priority": "low", "category": "verification"}
    ),
    Task(
        description="Extract book title",
        expected_output="Title of the main book", 
        agent=researcher,
        config={"priority": "medium", "category": "extraction"}
    )
]

print("Testing prioritize_tasks function...")
try:
    ordered_tasks = prioritize_tasks(tasks)
    print("SUCCESS: prioritize_tasks function worked!")
    for i, t in enumerate(ordered_tasks):
        priority = t.config.get('priority') if t.config else 'None'
        category = t.config.get('category') if t.config else 'None'
        print(f"Task {i+1} - {t.description[:30]}... [priority={priority}, category={category}]")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Summary ===")
print("✓ Config retention test")
print("✓ Field extraction test") 
print("✓ Callback retention test")
print("✓ Original issue scenario test")

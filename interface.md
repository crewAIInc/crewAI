# Public Interface

This document defines the public API surface that must remain stable and whose behavior must be preserved.

## File Handling Components

### FileHandler
**Location**: `lib/crewai/src/crewai/utilities/file_handler.py`

```python
class FileHandler:
    def __init__(self, file_path: bool | str) -> None:
        """Initialize with file path or boolean flag."""
        
    def log(self, **kwargs: Unpack[LogEntry]) -> None:
        """Log data with structured fields.
        
        Must be safe for concurrent calls from multiple threads/processes.
        """
```

**Required Behavior**:
- Concurrent `log()` calls must not corrupt the log file
- Each log entry must be written atomically
- JSON format must remain valid under concurrent writes
- File path initialization behavior must remain unchanged for users

### PickleHandler
**Location**: `lib/crewai/src/crewai/utilities/file_handler.py`

```python
class PickleHandler:
    def __init__(self, file_name: str) -> None:
        """Initialize with file name."""
        
    def save(self, data: Any) -> None:
        """Save data to pickle file.
        
        Must be safe for concurrent calls.
        """
        
    def load(self) -> Any:
        """Load data from pickle file.
        
        Must be safe for concurrent calls.
        """
```

**Required Behavior**:
- Concurrent `save()` operations must not corrupt the file
- `load()` must not fail due to concurrent `save()` operations
- File path behavior must remain unchanged for users

### TaskOutputStorageHandler
**Location**: `lib/crewai/src/crewai/utilities/task_output_storage_handler.py`

```python
class TaskOutputStorageHandler:
    def __init__(self) -> None:
        """Initialize storage handler."""
        
    def add(
        self,
        task: Task,
        output: dict[str, Any],
        task_index: int,
        inputs: dict[str, Any] | None = None,
        was_replayed: bool = False,
    ) -> None:
        """Add task output to storage.
        
        Must be safe for concurrent calls from multiple crews/threads.
        """
        
    def update(self, task_index: int, log: dict[str, Any]) -> None:
        """Update existing task output.
        
        Must be safe for concurrent calls.
        """
        
    def load(self) -> list[dict[str, Any]] | None:
        """Load all stored task outputs.
        
        Must be safe for concurrent calls.
        """
        
    def reset(self) -> None:
        """Clear all stored task outputs.
        
        Must be safe for concurrent calls.
        """
```

**Required Behavior**:
- Multiple crews running in parallel must have isolated storage OR proper locking
- Concurrent `add()` calls must not lose data or corrupt the database
- `load()` must return consistent data even during concurrent writes
- Each crew execution context must see its own task outputs

### KickoffTaskOutputsSQLiteStorage
**Location**: `lib/crewai/src/crewai/memory/storage/kickoff_task_outputs_storage.py`

```python
class KickoffTaskOutputsSQLiteStorage:
    def __init__(self, db_path: str | None = None) -> None:
        """Initialize SQLite storage with optional custom path."""
        
    def add(
        self,
        task: Task,
        output: dict[str, Any],
        task_index: int,
        was_replayed: bool = False,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Add task output record.
        
        Must be safe for concurrent calls.
        """
        
    def update(self, task_index: int, **kwargs: Any) -> None:
        """Update task output record.
        
        Must be safe for concurrent calls.
        """
        
    def load(self) -> list[dict[str, Any]]:
        """Load all task outputs.
        
        Must be safe for concurrent calls.
        """
        
    def delete_all(self) -> None:
        """Delete all task outputs.
        
        Must be safe for concurrent calls.
        """
```

**Required Behavior**:
- SQLite operations must handle concurrent access safely
- Database file must not become corrupted under concurrent writes
- Transactions must be properly isolated
- Default path behavior must provide isolation for concurrent executions

## Path Utilities

### db_storage_path
**Location**: `lib/crewai/src/crewai/utilities/paths.py`

```python
def db_storage_path() -> str:
    """Returns the path for SQLite database storage.
    
    Must return paths that don't collide under concurrent execution.
    """
```

**Required Behavior**:
- Concurrent executions must get isolated storage paths OR
- Returned path must be safe for concurrent database access
- Existing path resolution logic must remain compatible

## Task Output File Writing

### Task.output_file
**Location**: `lib/crewai/src/crewai/task.py`

The `Task` class supports writing outputs to files via the `output_file` parameter.

**Required Behavior**:
- Concurrent task executions with `output_file` set must not overwrite each other
- File writes must be atomic or properly synchronized
- Directory creation must be thread-safe

## Non-Public Implementation Details

The following are implementation details that may be modified as needed:
- Internal file locking mechanisms
- SQLite connection management strategies
- Temporary file usage for atomic writes
- Path generation algorithms for isolation
- Thread-local storage usage
- Process ID or thread ID incorporation into paths

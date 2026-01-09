# Advanced Tool Patterns

## Tool Composition

Create tools that work together as a cohesive toolkit:

```python
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Shared connection manager
class DatabaseToolMixin:
    """Mixin for tools that need database access."""

    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection

# Tool 1: Search
class ProjectSearchInput(BaseModel):
    query: str = Field(..., min_length=2)
    status: str = Field(default="all")

class ProjectSearchTool(DatabaseToolMixin, BaseTool):
    name: str = "project_search"
    description: str = """
    Search for projects by name or description.
    Returns project_id, name, status, and owner.
    Use project_id with 'project_get_details' for full info.
    """
    args_schema: Type[BaseModel] = ProjectSearchInput

    def _run(self, query: str, status: str = "all") -> str:
        results = self.db.search_projects(query, status)
        return json.dumps([{
            "project_id": p.id,
            "name": p.name,
            "status": p.status,
            "owner": p.owner_name
        } for p in results])

# Tool 2: Get Details (uses project_id from search)
class ProjectDetailsInput(BaseModel):
    project_id: str = Field(..., description="Project ID from search results")

class ProjectDetailsTool(DatabaseToolMixin, BaseTool):
    name: str = "project_get_details"
    description: str = """
    Get full project details by project_id.
    Use after 'project_search' to get comprehensive information.
    """
    args_schema: Type[BaseModel] = ProjectDetailsInput

    def _run(self, project_id: str) -> str:
        project = self.db.get_project(project_id)
        if not project:
            return f"Project '{project_id}' not found. Use 'project_search' to find valid IDs."
        return json.dumps({
            "project_id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status,
            "owner": project.owner_name,
            "team": [m.name for m in project.team],
            "milestones": [{
                "name": m.name,
                "due": m.due_date.isoformat(),
                "complete": m.is_complete
            } for m in project.milestones]
        })

# Tool 3: Update (uses project_id)
class ProjectUpdateInput(BaseModel):
    project_id: str = Field(...)
    status: str = Field(default=None)
    description: str = Field(default=None)

class ProjectUpdateTool(DatabaseToolMixin, BaseTool):
    name: str = "project_update"
    description: str = """
    Update project status or description.
    Requires project_id from search. At least one field must be provided.
    """
    args_schema: Type[BaseModel] = ProjectUpdateInput

    def _run(self, project_id: str, status: str = None, description: str = None) -> str:
        if not status and not description:
            return "No updates provided. Specify 'status' and/or 'description'."

        updates = {}
        if status:
            valid = ["active", "paused", "completed", "cancelled"]
            if status not in valid:
                return f"Invalid status. Choose from: {valid}"
            updates["status"] = status
        if description:
            updates["description"] = description

        self.db.update_project(project_id, **updates)
        return f"Project {project_id} updated successfully."
```

## Stateful Tools

Tools that maintain state across calls:

```python
class ConversationMemoryTool(BaseTool):
    name: str = "conversation_memory"
    description: str = """
    Store and retrieve conversation context.
    Actions: 'store' to save, 'recall' to retrieve, 'clear' to reset.
    """
    args_schema: Type[BaseModel] = MemoryInput

    def __init__(self):
        super().__init__()
        self._memory: dict[str, list] = {}

    def _run(self, action: str, key: str, value: str = None) -> str:
        if action == "store":
            if key not in self._memory:
                self._memory[key] = []
            self._memory[key].append({
                "value": value,
                "timestamp": datetime.now().isoformat()
            })
            return f"Stored under '{key}'"

        elif action == "recall":
            items = self._memory.get(key, [])
            if not items:
                return f"No memory found for '{key}'"
            return json.dumps(items)

        elif action == "clear":
            if key in self._memory:
                del self._memory[key]
                return f"Cleared '{key}'"
            return f"Nothing to clear for '{key}'"

        return f"Unknown action: {action}. Use 'store', 'recall', or 'clear'."
```

## Retry and Circuit Breaker Patterns

```python
from functools import wraps
import time

class ResilientAPITool(BaseTool):
    name: str = "api_call"
    description: str = "Make API calls with automatic retry"
    args_schema: Type[BaseModel] = APIInput

    def __init__(self, api_client, max_retries: int = 3):
        super().__init__()
        self.api = api_client
        self.max_retries = max_retries
        self._failures = 0
        self._circuit_open_until = None

    def _run(self, endpoint: str, params: dict = None) -> str:
        # Circuit breaker check
        if self._circuit_open_until:
            if datetime.now() < self._circuit_open_until:
                return "Service temporarily unavailable. Try again in 60 seconds."
            self._circuit_open_until = None
            self._failures = 0

        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self.api.call(endpoint, params)
                self._failures = 0  # Reset on success
                return json.dumps(result)
            except APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # Track failures for circuit breaker
        self._failures += 1
        if self._failures >= 5:
            self._circuit_open_until = datetime.now() + timedelta(seconds=60)

        return f"API call failed after {self.max_retries} attempts: {last_error}"
```

## Paginated Results Pattern

```python
class PaginatedSearchInput(BaseModel):
    query: str = Field(...)
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=25, ge=1, le=100)

class PaginatedSearchTool(BaseTool):
    name: str = "paginated_search"
    description: str = """
    Search with pagination. Returns results and pagination info.
    Use 'page' parameter to navigate through results.
    """
    args_schema: Type[BaseModel] = PaginatedSearchInput

    def _run(self, query: str, page: int = 1, per_page: int = 25) -> str:
        total, results = self.db.search_paginated(query, page, per_page)
        total_pages = (total + per_page - 1) // per_page

        return json.dumps({
            "results": results,
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_results": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        })
```

## File Operation Tools

```python
import os
from pathlib import Path

class FileReadInput(BaseModel):
    file_path: str = Field(..., description="Path relative to workspace root")
    encoding: str = Field(default="utf-8")

class SafeFileReadTool(BaseTool):
    name: str = "read_file"
    description: str = """
    Read file contents from the workspace.
    Only files within the workspace directory can be accessed.
    """
    args_schema: Type[BaseModel] = FileReadInput

    def __init__(self, workspace_root: str):
        super().__init__()
        self.workspace = Path(workspace_root).resolve()

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        # Security: prevent path traversal
        target = (self.workspace / file_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            return "Error: Access denied. Path must be within workspace."

        if not target.exists():
            return f"File not found: {file_path}"

        if not target.is_file():
            return f"Not a file: {file_path}"

        # Size limit
        if target.stat().st_size > 1_000_000:  # 1MB
            return "File too large (>1MB). Use 'read_file_chunk' for large files."

        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return f"Cannot read file with encoding '{encoding}'. Try 'latin-1' or 'binary'."
```

## Tool Testing Patterns

```python
import pytest
from unittest.mock import Mock, patch

class TestCustomerSearchTool:
    """Test suite for CustomerSearchTool."""

    @pytest.fixture
    def mock_db(self):
        db = Mock()
        db.search_customers.return_value = [
            Mock(id="C001", full_name="John Doe", email="john@example.com",
                 status="active", last_order_date=None)
        ]
        return db

    @pytest.fixture
    def tool(self, mock_db):
        return CustomerSearchTool(db_connection=mock_db)

    def test_basic_search(self, tool):
        result = tool._run(query="john")
        data = json.loads(result)
        assert data["count"] == 1
        assert data["customers"][0]["name"] == "John Doe"

    def test_no_results(self, tool, mock_db):
        mock_db.search_customers.return_value = []
        result = tool._run(query="nonexistent")
        assert "No customers found" in result

    def test_invalid_status(self, tool):
        result = tool._run(query="john", status="invalid")
        assert "Invalid status" in result

    def test_db_error_handling(self, tool, mock_db):
        mock_db.search_customers.side_effect = DatabaseError("Connection lost")
        result = tool._run(query="john")
        assert "Database unavailable" in result
```

## Structured Tool Output with Pydantic

```python
class AnalysisOutput(BaseModel):
    """Structured output for analysis tool."""
    summary: str
    confidence: float = Field(ge=0, le=1)
    key_findings: list[str]
    recommendations: list[str]

class AnalysisTool(BaseTool):
    name: str = "analyze_data"
    description: str = "Analyze data and return structured findings"
    args_schema: Type[BaseModel] = AnalysisInput

    def _run(self, data_source: str) -> str:
        raw_analysis = self.analyzer.analyze(data_source)

        # Structure the output
        output = AnalysisOutput(
            summary=raw_analysis.summary,
            confidence=raw_analysis.confidence_score,
            key_findings=raw_analysis.findings[:5],
            recommendations=raw_analysis.recommendations[:3]
        )

        return output.model_dump_json()
```

## Tool Initialization in Crews

```python
# In your crew file
from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from ..tools.customer_tools import CustomerSearchTool, CustomerDetailsTool
from ..tools.order_tools import OrderSearchTool

@CrewBase
class SupportCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, db_connection):
        super().__init__()
        # Initialize tools with shared dependencies
        self.customer_search = CustomerSearchTool(db_connection)
        self.customer_details = CustomerDetailsTool(db_connection)
        self.order_search = OrderSearchTool(db_connection)

    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["support_agent"],  # type: ignore[index]
            tools=[
                self.customer_search,
                self.customer_details,
                self.order_search
            ],
            verbose=True,
        )

    @task
    def resolve_inquiry(self) -> Task:
        return Task(config=self.tasks_config["resolve_inquiry"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks)
```

## Tool Caching

CrewAI caches tool results to avoid redundant calls. Use `cache_function` for fine-grained control.

### Cache Function Signature

```python
def cache_function(arguments: dict, result: Any) -> bool:
    """
    Determines whether to cache a result.

    Args:
        arguments: Dict of arguments passed to _run()
        result: The return value from _run()

    Returns:
        True to cache, False to skip caching
    """
    return True
```

### Always Cache (Default Behavior)

```python
from typing import Callable, Any

class ExpensiveAPITool(BaseTool):
    name: str = "expensive_api"
    description: str = "Call expensive API with caching"
    args_schema: Type[BaseModel] = APIInput

    # Cache all results
    cache_function: Callable[[dict, Any], bool] = lambda args, result: True

    def _run(self, query: str) -> str:
        return expensive_api.call(query)
```

### Conditional Caching - Only Cache Successes

```python
class DataFetchTool(BaseTool):
    name: str = "fetch_data"
    description: str = "Fetch data, only cache successful results"
    args_schema: Type[BaseModel] = FetchInput

    def _should_cache(self, args: dict, result: Any) -> bool:
        """Don't cache error responses."""
        if isinstance(result, str):
            return not result.startswith("Error:")
        return True

    cache_function: Callable[[dict, Any], bool] = _should_cache

    def _run(self, data_id: str) -> str:
        try:
            data = api.fetch(data_id)
            return json.dumps(data)
        except APIError as e:
            return f"Error: {e}"  # Won't be cached, agent can retry
```

### Conditional Caching - Based on Result Value

```python
class CalculationTool(BaseTool):
    name: str = "calculate"
    description: str = "Calculate with selective caching"
    args_schema: Type[BaseModel] = CalcInput

    # Only cache positive results
    cache_function: Callable[[dict, Any], bool] = lambda args, result: (
        isinstance(result, (int, float)) and result > 0
    )

    def _run(self, a: int, b: int) -> int:
        return a * b
```

### Conditional Caching - Based on Arguments

```python
class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search with argument-based caching"
    args_schema: Type[BaseModel] = SearchInput

    def _cache_strategy(self, args: dict, result: Any) -> bool:
        """Cache only broad searches, not user-specific ones."""
        # Don't cache user-specific searches (may change frequently)
        if args.get("user_id"):
            return False
        # Cache general searches
        return True

    cache_function: Callable[[dict, Any], bool] = _cache_strategy

    def _run(self, query: str, user_id: str = None) -> str:
        return db.search(query, user_id=user_id)
```

## Async Execution Patterns

**Important:** CrewAI tool `_run()` methods are synchronous. Async execution happens at the crew/task level.

### Tools Are Synchronous

```python
import requests

class FetchURLTool(BaseTool):
    name: str = "fetch_url"
    description: str = "Fetch content from URL"
    args_schema: Type[BaseModel] = URLInput

    def _run(self, url: str, timeout: int = 30) -> str:
        """Use synchronous HTTP client in _run()."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.Timeout:
            return f"Timeout after {timeout}s. Try increasing timeout."
        except requests.RequestException as e:
            return f"Request failed: {e}"
```

### Parallel External Calls Within a Tool

When a tool needs multiple independent external calls, use `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiSourceSearchTool(BaseTool):
    name: str = "multi_source_search"
    description: str = "Search multiple sources in parallel"
    args_schema: Type[BaseModel] = SearchInput

    def __init__(self, sources: list):
        super().__init__()
        self.sources = sources

    def _run(self, query: str) -> str:
        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            futures = {
                executor.submit(src.search, query): src.name
                for src in self.sources
            }

            results = {}
            for future in as_completed(futures):
                source_name = futures[future]
                try:
                    results[source_name] = future.result(timeout=10)
                except Exception as e:
                    results[source_name] = f"Error: {e}"

        return json.dumps(results)
```

### Async at Crew Level

Async execution is configured when kicking off crews:

```python
from crewai.flow.flow import Flow, listen

class ResearchFlow(Flow[ResearchState]):

    @listen(classify_topic)
    async def run_research_crew(self):
        crew = ResearchCrew().crew()

        # Option 1: Native async (preferred for high concurrency)
        result = await crew.akickoff(inputs={"topic": self.state.topic})

        # Option 2: Thread-based async
        result = await crew.kickoff_async(inputs={"topic": self.state.topic})

        self.state.research = result.raw
        return result

    @listen(run_research_crew)
    async def run_parallel_crews(self):
        # Run multiple crews concurrently
        crews = [
            AnalysisCrew().crew(),
            SummaryCrew().crew(),
            ValidationCrew().crew()
        ]

        results = await asyncio.gather(*[
            crew.akickoff(inputs={"data": self.state.research})
            for crew in crews
        ])

        return results
```

### Async Task Configuration

Mark tasks for async execution within a crew:

```python
# config/tasks.yaml
research_task:
  description: "Research the topic thoroughly"
  expected_output: "Comprehensive research findings"
  agent: researcher
  async_execution: true  # This task runs asynchronously

analysis_task:
  description: "Analyze research findings"
  expected_output: "Analysis report"
  agent: analyst
  context:
    - research_task  # Waits for research_task to complete
```

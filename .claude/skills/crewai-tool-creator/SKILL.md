---
name: crewai-tool-creator
description: |
  Guide for creating custom tools for CrewAI agents following best practices from both CrewAI documentation
  and Anthropic's agent tool design principles. Use this skill when: (1) Creating new custom tools for agents,
  (2) Designing tool input schemas with Pydantic, (3) Writing effective tool descriptions and error messages,
  (4) Implementing caching and async tools, (5) Optimizing tools for context efficiency and token consumption.
  Always use BaseTool class inheritance for full control over validation, error handling, and behavior.
---

# CrewAI Tool Creator Guide

**Core Principle: Design tools that reduce agent cognitive load while enabling clear, distinct actions.**

Tools are how agents interact with the world. Well-designed tools make agents more effective; poorly designed tools cause confusion, hallucinations, and wasted tokens.

## BaseTool Class Structure

Always use the `BaseTool` class for full control over input validation, error handling, and state:

```python
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchContactsInput(BaseModel):
    """Input schema for contact search."""
    query: str = Field(..., description="Name or email to search for (partial match)")
    limit: int = Field(default=10, ge=1, le=100, description="Max results (1-100)")

class SearchContactsTool(BaseTool):
    name: str = "search_contacts"
    description: str = """
    Search for contacts by name or email. Returns matching contacts with
    their name, email, and role. Use this instead of listing all contacts.
    """
    args_schema: Type[BaseModel] = SearchContactsInput

    def _run(self, query: str, limit: int = 10) -> str:
        try:
            results = self.db.search_contacts(query, limit)
            return json.dumps(results)
        except DatabaseError as e:
            return f"Search failed: {e}. Try a different query or check connection."
```

## Critical Design Principles

### 1. Context Efficiency

Agents have limited context. Design tools that consolidate operations:

```python
# BAD: Forces multiple calls, wastes context
class ListContactsTool(BaseTool):
    name: str = "list_contacts"
    description: str = "List all contacts in the database."

    def _run(self) -> str:
        return json.dumps(db.get_all())  # Could return thousands

# GOOD: Search reduces context consumption
class SearchContactsTool(BaseTool):
    name: str = "search_contacts"
    description: str = "Search contacts by name/email. Max 25 results per query."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, limit: int = 25) -> str:
        return json.dumps(db.search(query, limit))
```

### 2. Clear Purpose and Naming

Each tool should have one clear purpose with an unambiguous name:

```python
# BAD: Vague, what does "process" mean?
name: str = "data_tool"

# GOOD: Clear action and target
name: str = "search_customer_orders"

# Use namespacing for related tools
name: str = "crm_search_contacts"
name: str = "crm_create_contact"
name: str = "crm_update_contact"
```

### 3. Semantic Parameter Names

Use descriptive names, not cryptic identifiers:

```python
# BAD: Ambiguous
class BadInput(BaseModel):
    user: str  # User what? ID? Name? Email?
    id: str    # ID of what?

# GOOD: Unambiguous
class GoodInput(BaseModel):
    user_email: str = Field(..., description="Email address of the user")
    order_id: str = Field(..., description="Order ID (format: ORD-XXXXX)")
```

### 4. Meaningful Response Design

Return what agents need, exclude what they don't:

```python
# BAD: Dumps everything including useless metadata
def _run(self, order_id: str) -> str:
    order = db.get_order(order_id)
    return json.dumps(order.__dict__)  # Includes uuid, created_at_unix, internal_flags...

# GOOD: Curated, relevant fields
def _run(self, order_id: str) -> str:
    order = db.get_order(order_id)
    return json.dumps({
        "order_id": order.display_id,
        "customer": order.customer_name,
        "items": [{"name": i.name, "qty": i.qty} for i in order.items],
        "status": order.status,
        "total": f"${order.total:.2f}"
    })
```

### 5. Actionable Error Messages

Replace stack traces with guidance:

```python
# BAD: Unhelpful
def _run(self, date: str) -> str:
    try:
        parsed = datetime.fromisoformat(date)
    except ValueError:
        return "Error: Invalid date format"

# GOOD: Guides correction
def _run(self, date: str) -> str:
    try:
        parsed = datetime.fromisoformat(date)
    except ValueError:
        return (
            f"Invalid date format: '{date}'. "
            f"Use ISO 8601 format: YYYY-MM-DD (e.g., 2024-03-15)"
        )
```

## Tool Description Best Practices

Write descriptions as you would for a new team member:

```python
description: str = """
Search for orders by customer email or order ID.

When to use:
- Finding a specific customer's order history
- Looking up order status by ID
- Checking recent orders for a customer

Returns: Order details including status, items, and total.
Does NOT return: Payment details or internal notes.

Tip: For bulk order analysis, use 'export_orders' instead.
"""
```

## Pydantic Input Schema Patterns

### Required vs Optional Fields

```python
class AnalyzeInput(BaseModel):
    # Required: no default value
    data_source: str = Field(..., description="Data source identifier")

    # Optional with default
    include_historical: bool = Field(
        default=False,
        description="Include historical data (slower)"
    )

    # Optional, can be None
    date_filter: Optional[str] = Field(
        default=None,
        description="Filter by date (ISO format) or None for all"
    )
```

### Validation Constraints

```python
class PaginatedInput(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=25, ge=1, le=100, description="Results per page")

    # Complex validation
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if v.strip() != v:
            raise ValueError("Query cannot have leading/trailing whitespace")
        return v
```

### Enum for Controlled Values

```python
from enum import Enum

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"

class SortableInput(BaseModel):
    query: str = Field(...)
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: SortOrder = Field(default=SortOrder.DESC)
```

## Response Format Flexibility

Let agents request the detail level they need:

```python
class AnalyzeDataInput(BaseModel):
    data_source: str = Field(..., description="Data source identifier")
    response_format: str = Field(
        default="concise",
        description="'concise' for summary only, 'detailed' for full breakdown"
    )

class AnalyzeDataTool(BaseTool):
    name: str = "analyze_data"
    description: str = "Analyze data source. Use response_format='concise' unless full details needed."
    args_schema: Type[BaseModel] = AnalyzeDataInput

    def _run(self, data_source: str, response_format: str = "concise") -> str:
        analysis = self.analyze(data_source)

        if response_format == "concise":
            return json.dumps({
                "summary": analysis.summary,
                "key_metrics": analysis.top_3_metrics
            })  # ~70 tokens
        else:
            return json.dumps({
                "summary": analysis.summary,
                "all_metrics": analysis.all_metrics,
                "data_points": analysis.raw_data,
                "methodology": analysis.methodology
            })  # ~300 tokens
```

## Tool with Constructor Dependencies

Inject dependencies through `__init__`:

```python
class DatabaseSearchTool(BaseTool):
    name: str = "db_search"
    description: str = "Search the database"
    args_schema: Type[BaseModel] = SearchInput

    def __init__(self, db_connection, cache_client=None):
        super().__init__()
        self.db = db_connection
        self.cache = cache_client

    def _run(self, query: str) -> str:
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                return cached

        results = self.db.search(query)
        if self.cache:
            self.cache.set(query, results)
        return results
```

## Tool Caching

**When to use:** Expensive API calls, idempotent operations, repeated queries with same results.

Use the `cache_function` attribute to control caching. It receives `(arguments: dict, result)` and returns `bool`.

```python
from typing import Callable, Any

class ExpensiveSearchTool(BaseTool):
    name: str = "expensive_search"
    description: str = "Search with caching for repeated queries"
    args_schema: Type[BaseModel] = SearchInput

    # Cache all successful results
    cache_function: Callable[[dict, Any], bool] = lambda args, result: (
        not str(result).startswith("Error:")  # Don't cache errors
    )

    def _run(self, query: str) -> str:
        return expensive_api.search(query)
```

**When NOT to cache:**
- User-specific data that changes frequently
- Time-sensitive information
- Error responses (agent should retry)

See [references/tool-patterns.md](references/tool-patterns.md#tool-caching) for conditional caching patterns.

## Async Execution

**Important:** Tool `_run()` methods are **synchronous**. Async execution happens at the crew/task level, not the tool level.

**When you need parallel operations within a tool**, use `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiSourceTool(BaseTool):
    def _run(self, query: str) -> str:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(src.search, query): src for src in self.sources}
            results = {futures[f]: f.result() for f in as_completed(futures)}
        return json.dumps(results)
```

**For async crew execution**, use `akickoff()` or `kickoff_async()` at the flow level.

See [references/tool-patterns.md](references/tool-patterns.md#async-execution-patterns) for async patterns.

## Complete Production Example

```python
from typing import Type, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json

class CustomerSearchInput(BaseModel):
    """Input schema for customer search tool."""
    query: str = Field(
        ...,
        min_length=2,
        description="Search by name, email, or phone (min 2 chars)"
    )
    status: Optional[str] = Field(
        default=None,
        description="Filter by status: 'active', 'inactive', or None for all"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Results per page (1-50, default 10)"
    )

class CustomerSearchTool(BaseTool):
    name: str = "crm_search_customers"
    description: str = """
    Search for customers by name, email, or phone number.

    Use cases:
    - Find a specific customer's profile
    - Look up customers by partial name/email
    - Filter active vs inactive customers

    Returns: Customer name, email, status, and last order date.
    For full customer details, use 'crm_get_customer' with the customer_id.
    """
    args_schema: Type[BaseModel] = CustomerSearchInput

    def __init__(self, db_connection):
        super().__init__()
        self.db = db_connection

    def _run(self, query: str, status: Optional[str] = None, limit: int = 10) -> str:
        # Validate status if provided
        valid_statuses = {"active", "inactive", None}
        if status and status not in valid_statuses:
            return (
                f"Invalid status: '{status}'. "
                f"Use 'active', 'inactive', or omit for all customers."
            )

        try:
            results = self.db.search_customers(
                query=query,
                status=status,
                limit=limit
            )

            if not results:
                return f"No customers found matching '{query}'" + (
                    f" with status '{status}'" if status else ""
                ) + ". Try a broader search term."

            # Return curated, agent-friendly format
            return json.dumps({
                "count": len(results),
                "customers": [
                    {
                        "customer_id": c.id,
                        "name": c.full_name,
                        "email": c.email,
                        "status": c.status,
                        "last_order": c.last_order_date.isoformat() if c.last_order_date else None
                    }
                    for c in results
                ]
            })

        except DatabaseError:
            return "Search failed: Database unavailable. Retry in a moment or contact support."
```

## Assigning Tools to Agents

```python
from crewai import Agent

# Initialize tools with dependencies
search_tool = CustomerSearchTool(db_connection=db)
order_tool = OrderLookupTool(db_connection=db)

# Assign to agent
support_agent = Agent(
    role="Customer Support Specialist",
    goal="Quickly resolve customer inquiries",
    backstory="Expert at navigating customer systems",
    tools=[search_tool, order_tool],
    verbose=True
)
```

## Reference Files

- [references/tool-patterns.md](references/tool-patterns.md) - Advanced patterns: MCP integration, tool composition, testing
- [references/design-principles.md](references/design-principles.md) - Anthropic's complete agent tool design principles

## Sources

- [CrewAI Custom Tools Documentation](https://docs.crewai.com/en/learn/create-custom-tools)
- [Anthropic: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

# Agent Tool Design Principles

Based on [Anthropic's Engineering Guide: Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

## Core Philosophy

> "LLM agents have limited 'context'...whereas computer memory is cheap and abundant"

Agents operate under fundamentally different constraints than traditional software. Every token consumed reduces their ability to reason. Design tools that maximize value per token.

## Principle 1: Context Efficiency

### The Problem
Agents have limited context windows. Every tool response consumes tokens that could be used for reasoning.

### The Solution
Build tools that consolidate operations and return only what's needed.

```python
# ANTI-PATTERN: List everything
class ListAllContactsTool(BaseTool):
    name: str = "list_contacts"

    def _run(self) -> str:
        # Returns 10,000 contacts = 500,000 tokens wasted
        return json.dumps(db.get_all_contacts())

# PATTERN: Search with limits
class SearchContactsTool(BaseTool):
    name: str = "search_contacts"
    description: str = "Search contacts. Use specific queries to find who you need."

    def _run(self, query: str, limit: int = 25) -> str:
        # Returns max 25 results = ~2,000 tokens
        return json.dumps(db.search(query, limit=limit))
```

### Multi-Step Consolidation

Combine related operations that are typically used together:

```python
# ANTI-PATTERN: Requires 3 tool calls
# 1. get_user(id) -> user
# 2. get_user_orders(user_id) -> orders
# 3. get_order_details(order_id) -> details

# PATTERN: Single call with context
class UserContextTool(BaseTool):
    name: str = "get_user_context"
    description: str = """
    Get user profile with recent activity.
    Returns user info, last 5 orders, and active support tickets.
    """

    def _run(self, user_id: str) -> str:
        user = db.get_user(user_id)
        orders = db.get_recent_orders(user_id, limit=5)
        tickets = db.get_open_tickets(user_id)

        return json.dumps({
            "user": {"name": user.name, "email": user.email, "tier": user.tier},
            "recent_orders": [{"id": o.id, "date": o.date, "status": o.status} for o in orders],
            "open_tickets": len(tickets)
        })
```

## Principle 2: Clear, Distinct Purpose

### Tool Selection
Agents must choose between available tools. Overlapping purposes cause confusion.

```python
# ANTI-PATTERN: Overlapping tools
tools = [
    DataTool(),      # "Process data"
    AnalyzerTool(),  # "Analyze data"
    ProcessorTool(), # "Handle data processing"
]
# Agent: "Which one do I use?"

# PATTERN: Distinct purposes
tools = [
    DataValidationTool(),  # "Validate data format and completeness"
    DataTransformTool(),   # "Convert data between formats (CSV, JSON, XML)"
    DataAnalysisTool(),    # "Calculate statistics and identify patterns"
]
```

### Namespacing

Group related tools with consistent prefixes:

```python
# Service-based namespacing
name: str = "asana_search_tasks"
name: str = "asana_create_task"
name: str = "asana_update_task"

# Or resource-based
name: str = "search_asana_tasks"
name: str = "create_asana_task"
name: str = "update_asana_task"
```

## Principle 3: Semantic Clarity

### Parameter Naming

Agents handle natural language better than cryptic identifiers:

```python
# ANTI-PATTERN: Ambiguous parameters
class BadInput(BaseModel):
    user: str      # ID? Email? Name?
    id: str        # Of what?
    type: str      # What types exist?

# PATTERN: Self-documenting parameters
class GoodInput(BaseModel):
    user_email: str = Field(..., description="User's email address")
    project_id: str = Field(..., description="Project ID (format: PRJ-XXXXX)")
    task_type: str = Field(
        ...,
        description="Type of task: 'bug', 'feature', or 'improvement'"
    )
```

### ID Resolution

Resolve UUIDs to semantic identifiers:

```python
# ANTI-PATTERN: Returns raw UUIDs
def _run(self, query: str) -> str:
    return json.dumps([{
        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "parent_id": "f1e2d3c4-b5a6-0987-dcba-654321098765"
    }])

# PATTERN: Semantic identifiers
def _run(self, query: str) -> str:
    return json.dumps([{
        "project_id": "PRJ-001",
        "project_name": "Website Redesign",
        "parent_project": "Marketing Initiatives"
    }])
```

## Principle 4: Meaningful Response Design

### Curated Fields

Return what enables action, exclude noise:

```python
# ANTI-PATTERN: Dump everything
def _run(self, order_id: str) -> str:
    order = db.get_order(order_id)
    return json.dumps({
        "uuid": order.uuid,
        "created_at_unix": order.created_at_unix,
        "updated_at_unix": order.updated_at_unix,
        "internal_flags": order.internal_flags,
        "mime_type": order.mime_type,
        "256px_image_url": order.thumbnail_url,
        "customer_name": order.customer_name,
        "status": order.status,
        # ... 50 more fields
    })

# PATTERN: Agent-relevant fields only
def _run(self, order_id: str) -> str:
    order = db.get_order(order_id)
    return json.dumps({
        "order_id": order.display_id,
        "customer": order.customer_name,
        "status": order.status,
        "items": [{"name": i.name, "qty": i.qty} for i in order.items],
        "total": f"${order.total:.2f}",
        "can_modify": order.status in ["pending", "processing"]
    })
```

### Response Format Flexibility

Let agents request appropriate detail levels:

```python
class AnalysisInput(BaseModel):
    target: str
    response_format: str = Field(
        default="concise",
        description="'concise' (~50 tokens) or 'detailed' (~500 tokens)"
    )

def _run(self, target: str, response_format: str = "concise") -> str:
    analysis = self.analyze(target)

    if response_format == "concise":
        return json.dumps({
            "summary": analysis.summary,
            "score": analysis.score
        })
    else:
        return json.dumps({
            "summary": analysis.summary,
            "score": analysis.score,
            "breakdown": analysis.category_scores,
            "evidence": analysis.supporting_data,
            "methodology": analysis.methodology_notes
        })
```

## Principle 5: Actionable Error Messages

### Replace Tracebacks with Guidance

```python
# ANTI-PATTERN: Opaque errors
def _run(self, date: str) -> str:
    try:
        dt = datetime.fromisoformat(date)
    except:
        return "ValueError: Invalid isoformat string"

# PATTERN: Corrective guidance
def _run(self, date: str) -> str:
    try:
        dt = datetime.fromisoformat(date)
    except ValueError:
        return (
            f"Invalid date format: '{date}'. "
            f"Expected ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS. "
            f"Example: 2024-03-15 or 2024-03-15T14:30:00"
        )
```

### Suggest Alternatives

```python
def _run(self, status: str) -> str:
    valid_statuses = ["pending", "active", "completed", "cancelled"]

    if status not in valid_statuses:
        # Find close matches
        close = [s for s in valid_statuses if status.lower() in s]
        suggestion = f" Did you mean '{close[0]}'?" if close else ""
        return f"Invalid status: '{status}'.{suggestion} Valid options: {valid_statuses}"
```

### Guide Efficient Behavior

```python
def _run(self, query: str, limit: int = 25) -> str:
    results = db.search(query, limit=limit)

    if len(results) == limit:
        return json.dumps({
            "results": results,
            "note": f"Returned max {limit} results. Refine query for better matches."
        })

    return json.dumps({"results": results})
```

## Principle 6: Effective Descriptions

### Write for a New Team Member

```python
description: str = """
Search for customer support tickets by keyword or status.

When to use:
- Finding tickets about a specific issue
- Checking status of customer-reported problems
- Looking up tickets by customer email

Parameters:
- query: Search term (searches subject and body)
- status: Filter by 'open', 'pending', 'resolved', or 'all'
- assigned_to: Filter by agent email (optional)

Returns: Ticket ID, subject, status, customer email, last update.
For full ticket details including conversation history, use 'get_ticket_details'.

Note: Only returns last 30 days by default. Use date_range for older tickets.
"""
```

### Clarify Relationships

```python
description: str = """
Create a new task in a project.

Requires:
- project_id: Get from 'search_projects' or 'list_my_projects'
- assignee_id: Get from 'search_team_members' (optional)

Creates task and returns task_id for use with:
- 'add_task_comment': Add notes or updates
- 'update_task_status': Change status
- 'add_task_attachment': Attach files
"""
```

## Principle 7: Truncation and Limits

### Default Limits

```python
DEFAULT_LIMIT = 25
MAX_RESPONSE_TOKENS = 25000

def _run(self, query: str, limit: int = DEFAULT_LIMIT) -> str:
    results = db.search(query, limit=min(limit, 100))

    response = json.dumps(results)

    # Enforce token limit
    if len(response) > MAX_RESPONSE_TOKENS * 4:  # ~4 chars per token
        return json.dumps({
            "results": results[:10],
            "truncated": True,
            "total_available": len(results),
            "suggestion": "Refine your query for more specific results"
        })

    return response
```

### Pagination Guidance

```python
def _run(self, query: str, page: int = 1) -> str:
    per_page = 25
    total, results = db.search_paginated(query, page, per_page)

    return json.dumps({
        "results": results,
        "page": page,
        "total_pages": (total + per_page - 1) // per_page,
        "hint": "Use 'page' parameter for more results" if total > per_page else None
    })
```

## Principle 8: Evaluation-Driven Development

### Build Real-World Evaluations

Test tools with realistic, multi-step scenarios:

```python
# Evaluation scenarios
EVAL_SCENARIOS = [
    {
        "name": "customer_lookup_flow",
        "prompt": "Find John Smith's last order and check if it shipped",
        "expected_tools": ["search_customers", "get_order_details"],
        "success_criteria": lambda result: "shipped" in result.lower() or "pending" in result.lower()
    },
    {
        "name": "refund_processing",
        "prompt": "Process a refund for order ORD-12345",
        "expected_tools": ["get_order_details", "process_refund"],
        "success_criteria": lambda result: "refund" in result.lower() and "processed" in result.lower()
    }
]
```

### Metrics to Track

- **Accuracy**: Did the agent complete the task correctly?
- **Tool calls**: How many calls were needed? (fewer is better)
- **Token consumption**: Total tokens used
- **Error rate**: How often did tools return errors?
- **Recovery rate**: Did the agent recover from errors?

### Iterate with Agent Feedback

After evaluations, let agents analyze their own transcripts:

```
Analyze this tool usage transcript. Identify:
1. Where did I get confused about which tool to use?
2. Which tool descriptions were unclear?
3. What information was missing from tool responses?
4. Where did I make unnecessary tool calls?
```

## Summary Checklist

Before deploying a tool, verify:

- [ ] **Context Efficient**: Returns only necessary data
- [ ] **Clear Purpose**: Distinct from other tools, unambiguous name
- [ ] **Semantic Names**: Parameters are self-documenting
- [ ] **Meaningful Responses**: Curated fields, no noise
- [ ] **Actionable Errors**: Guide correction, suggest alternatives
- [ ] **Good Description**: Written for a new team member
- [ ] **Reasonable Limits**: Default pagination, token limits
- [ ] **Tested**: Evaluated with realistic multi-step scenarios

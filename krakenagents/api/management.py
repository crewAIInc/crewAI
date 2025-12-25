"""CrewManager REST API endpoints for managing agents, crews, tasks, and tools.

This module exposes the CrewManager CRUD operations via FastAPI endpoints,
allowing the dashboard UI to manage CrewAI configurations.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from crewai.manager import (
    CrewManager,
    AgentConfig,
    TaskConfig,
    CrewConfig,
    ToolConfig,
)
from crewai.manager.storage import SQLiteStorage

# Initialize manager with SQLite storage for persistence
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

manager = CrewManager(storage=SQLiteStorage(str(DATA_DIR / "crews.db")))

router = APIRouter(prefix="/api/management", tags=["management"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateAgentRequest(BaseModel):
    """Request body for creating an agent."""
    role: str = Field(..., description="The role of the agent")
    goal: str = Field(..., description="The objective of the agent")
    backstory: str = Field(..., description="Background context for the agent")
    name: str | None = Field(default=None, description="Display name")
    llm: str | None = Field(default=None, description="LLM model identifier")
    tool_ids: list[str] = Field(default_factory=list, description="Tool IDs")
    max_iter: int = Field(default=25, description="Maximum iterations")
    max_rpm: int | None = Field(default=None, description="Max requests per minute")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    allow_delegation: bool = Field(default=False, description="Allow delegation")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class UpdateAgentRequest(BaseModel):
    """Request body for updating an agent."""
    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    name: str | None = None
    llm: str | None = None
    tool_ids: list[str] | None = None
    max_iter: int | None = None
    max_rpm: int | None = None
    verbose: bool | None = None
    allow_delegation: bool | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class CreateTaskRequest(BaseModel):
    """Request body for creating a task."""
    description: str = Field(..., description="Task description")
    expected_output: str = Field(default="", description="Expected output")
    name: str | None = Field(default=None, description="Task name")
    agent_id: str | None = Field(default=None, description="Assigned agent ID")
    agent_role: str | None = Field(default=None, description="Agent role (alternative)")
    tool_ids: list[str] = Field(default_factory=list, description="Available tools")
    context_task_ids: list[str] = Field(default_factory=list, description="Context tasks")
    async_execution: bool = Field(default=False, description="Run async")
    human_input: bool = Field(default=False, description="Require human input")
    action_based: bool = Field(default=True, description="Action-based task")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class UpdateTaskRequest(BaseModel):
    """Request body for updating a task."""
    description: str | None = None
    expected_output: str | None = None
    name: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None
    tool_ids: list[str] | None = None
    context_task_ids: list[str] | None = None
    async_execution: bool | None = None
    human_input: bool | None = None
    action_based: bool | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class CreateCrewRequest(BaseModel):
    """Request body for creating a crew."""
    name: str = Field(..., description="Crew name")
    description: str = Field(default="", description="Crew description")
    agent_ids: list[str] = Field(default_factory=list, description="Agent IDs")
    task_ids: list[str] = Field(default_factory=list, description="Task IDs")
    process: str = Field(default="sequential", description="Process type")
    manager_agent_id: str | None = Field(default=None, description="Manager agent")
    manager_llm: str | None = Field(default=None, description="Manager LLM")
    verbose: bool = Field(default=False, description="Verbose mode")
    memory: bool = Field(default=False, description="Enable memory")
    cache: bool = Field(default=True, description="Enable cache")
    stream: bool = Field(default=True, description="Enable streaming")
    max_rpm: int | None = Field(default=None, description="Max requests per minute")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class UpdateCrewRequest(BaseModel):
    """Request body for updating a crew."""
    name: str | None = None
    description: str | None = None
    agent_ids: list[str] | None = None
    task_ids: list[str] | None = None
    process: str | None = None
    manager_agent_id: str | None = None
    manager_llm: str | None = None
    verbose: bool | None = None
    memory: bool | None = None
    cache: bool | None = None
    stream: bool | None = None
    max_rpm: int | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class CreateToolRequest(BaseModel):
    """Request body for creating/registering a tool."""
    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    tool_type: str = Field(default="builtin", description="Tool type")
    class_name: str | None = Field(default=None, description="Tool class name")
    module_path: str | None = Field(default=None, description="Module path")
    init_kwargs: dict[str, Any] = Field(default_factory=dict, description="Init kwargs")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment vars")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class UpdateToolRequest(BaseModel):
    """Request body for updating a tool."""
    name: str | None = None
    description: str | None = None
    tool_type: str | None = None
    class_name: str | None = None
    module_path: str | None = None
    init_kwargs: dict[str, Any] | None = None
    env_vars: dict[str, str] | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ExecuteCrewRequest(BaseModel):
    """Request body for executing a crew."""
    inputs: dict[str, Any] = Field(default_factory=dict, description="Crew inputs")


# =============================================================================
# Agent Endpoints
# =============================================================================


@router.post("/agents")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent configuration."""
    config = AgentConfig(
        role=request.role,
        goal=request.goal,
        backstory=request.backstory,
        name=request.name,
        llm=request.llm,
        tool_ids=request.tool_ids,
        max_iter=request.max_iter,
        max_rpm=request.max_rpm,
        verbose=request.verbose,
        allow_delegation=request.allow_delegation,
        tags=request.tags,
        metadata=request.metadata,
    )
    result = manager.create_agent(config)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "data": result.data.model_dump(), "timestamp": result.timestamp.isoformat()}


@router.get("/agents")
async def list_agents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    tags: str | None = Query(default=None, description="Comma-separated tags"),
):
    """List all agents with pagination."""
    tag_list = tags.split(",") if tags else None
    result = manager.list_agents(offset=offset, limit=limit, tags=tag_list)
    return {
        "items": [a.model_dump() for a in result.items],
        "total": result.total,
        "offset": result.offset,
        "limit": result.limit,
        "hasMore": result.has_more,
    }


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get an agent by ID."""
    result = manager.get_agent(agent_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, request: UpdateAgentRequest):
    """Update an agent configuration."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = manager.update_agent(agent_id, updates)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent by ID."""
    result = manager.delete_agent(agent_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "deleted": True}


# =============================================================================
# Task Endpoints
# =============================================================================


@router.post("/tasks")
async def create_task(request: CreateTaskRequest):
    """Create a new task configuration."""
    config = TaskConfig(
        description=request.description,
        expected_output=request.expected_output,
        name=request.name,
        agent_id=request.agent_id,
        agent_role=request.agent_role,
        tool_ids=request.tool_ids,
        context_task_ids=request.context_task_ids,
        async_execution=request.async_execution,
        human_input=request.human_input,
        action_based=request.action_based,
        tags=request.tags,
        metadata=request.metadata,
    )
    result = manager.create_task(config)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "data": result.data.model_dump(), "timestamp": result.timestamp.isoformat()}


@router.get("/tasks")
async def list_tasks(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    tags: str | None = Query(default=None, description="Comma-separated tags"),
):
    """List all tasks with pagination."""
    tag_list = tags.split(",") if tags else None
    result = manager.list_tasks(offset=offset, limit=limit, tags=tag_list)
    return {
        "items": [t.model_dump() for t in result.items],
        "total": result.total,
        "offset": result.offset,
        "limit": result.limit,
        "hasMore": result.has_more,
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get a task by ID."""
    result = manager.get_task(task_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, request: UpdateTaskRequest):
    """Update a task configuration."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = manager.update_task(task_id, updates)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task by ID."""
    result = manager.delete_task(task_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "deleted": True}


# =============================================================================
# Crew Endpoints
# =============================================================================


@router.post("/crews")
async def create_crew(request: CreateCrewRequest):
    """Create a new crew configuration."""
    config = CrewConfig(
        name=request.name,
        description=request.description,
        agent_ids=request.agent_ids,
        task_ids=request.task_ids,
        process=request.process,
        manager_agent_id=request.manager_agent_id,
        manager_llm=request.manager_llm,
        verbose=request.verbose,
        memory=request.memory,
        cache=request.cache,
        stream=request.stream,
        max_rpm=request.max_rpm,
        tags=request.tags,
        metadata=request.metadata,
    )
    result = manager.create_crew(config)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "data": result.data.model_dump(), "timestamp": result.timestamp.isoformat()}


@router.get("/crews")
async def list_crews(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    tags: str | None = Query(default=None, description="Comma-separated tags"),
):
    """List all crews with pagination."""
    tag_list = tags.split(",") if tags else None
    result = manager.list_crews(offset=offset, limit=limit, tags=tag_list)
    return {
        "items": [c.model_dump() for c in result.items],
        "total": result.total,
        "offset": result.offset,
        "limit": result.limit,
        "hasMore": result.has_more,
    }


@router.get("/crews/{crew_id}")
async def get_crew(crew_id: str):
    """Get a crew by ID."""
    result = manager.get_crew(crew_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.put("/crews/{crew_id}")
async def update_crew(crew_id: str, request: UpdateCrewRequest):
    """Update a crew configuration."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = manager.update_crew(crew_id, updates)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.delete("/crews/{crew_id}")
async def delete_crew(crew_id: str):
    """Delete a crew by ID."""
    result = manager.delete_crew(crew_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "deleted": True}


@router.post("/crews/{crew_id}/execute")
async def execute_crew(crew_id: str, request: ExecuteCrewRequest):
    """Execute a crew by ID."""
    result = manager.execute_crew(crew_id, inputs=request.inputs)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {
        "success": True,
        "crewId": result.crew_id,
        "rawOutput": result.raw_output,
        "taskOutputs": [t.model_dump() for t in result.task_outputs],
        "tokenUsage": result.token_usage,
        "executionTimeSeconds": result.execution_time_seconds,
    }


# =============================================================================
# Tool Endpoints
# =============================================================================


@router.post("/tools")
async def create_tool(request: CreateToolRequest):
    """Create/register a new tool configuration."""
    config = ToolConfig(
        name=request.name,
        description=request.description,
        tool_type=request.tool_type,
        class_name=request.class_name,
        module_path=request.module_path,
        init_kwargs=request.init_kwargs,
        env_vars=request.env_vars,
        tags=request.tags,
        metadata=request.metadata,
    )
    result = manager.create_tool(config)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "data": result.data.model_dump(), "timestamp": result.timestamp.isoformat()}


@router.get("/tools")
async def list_tools(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    tags: str | None = Query(default=None, description="Comma-separated tags"),
):
    """List all tools with pagination."""
    tag_list = tags.split(",") if tags else None
    result = manager.list_tools(offset=offset, limit=limit, tags=tag_list)
    return {
        "items": [t.model_dump() for t in result.items],
        "total": result.total,
        "offset": result.offset,
        "limit": result.limit,
        "hasMore": result.has_more,
    }


@router.get("/tools/{tool_id}")
async def get_tool(tool_id: str):
    """Get a tool by ID."""
    result = manager.get_tool(tool_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.put("/tools/{tool_id}")
async def update_tool(tool_id: str, request: UpdateToolRequest):
    """Update a tool configuration."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    result = manager.update_tool(tool_id, updates)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "data": result.data.model_dump()}


@router.delete("/tools/{tool_id}")
async def delete_tool(tool_id: str):
    """Delete a tool by ID."""
    result = manager.delete_tool(tool_id)
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    return {"success": True, "deleted": True}


# =============================================================================
# Utility Endpoints
# =============================================================================


@router.get("/builtin-tools")
async def list_builtin_tools():
    """List available built-in tools from crewai_tools."""
    try:
        import crewai_tools
        tools = []
        for name in dir(crewai_tools):
            if name.endswith("Tool") and not name.startswith("_"):
                tool_class = getattr(crewai_tools, name)
                if hasattr(tool_class, "__doc__"):
                    tools.append({
                        "name": name,
                        "description": tool_class.__doc__ or "",
                        "module": "crewai_tools",
                    })
        return {"tools": tools}
    except ImportError:
        return {"tools": [], "error": "crewai_tools not installed"}


@router.post("/export")
async def export_data(format: str = Query(default="json", enum=["json", "yaml"])):
    """Export all configuration data."""
    export_path = DATA_DIR / f"export.{format}"
    result = manager.export_to_file(str(export_path), format=format)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {"success": True, "path": result.data}


@router.post("/import")
async def import_data(file_path: str = Query(..., description="Path to import file")):
    """Import configuration data from a file."""
    result = manager.import_from_file(file_path)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return {"success": True, "imported": True}


@router.post("/clear")
async def clear_all_data():
    """Clear all configuration data (use with caution!)."""
    result = manager.clear_all()
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {"success": True, "cleared": True}

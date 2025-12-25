"""FastAPI WebSocket server for QRI Trading Dashboard.

Provides real-time streaming of agent activities to the frontend.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from krakenagents.api.management import router as management_router


# =============================================================================
# Models
# =============================================================================

class StreamChunk(BaseModel):
    """Message format for WebSocket streaming."""
    content: str
    chunk_type: str  # text, tool_call, task_started, task_completed, error, heartbeat
    task_id: str = ""
    task_name: str = ""
    task_index: int = 0
    agent_role: str = ""
    agent_id: str = ""
    timestamp: str = ""
    tool_call: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}

    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        super().__init__(**data)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override to output camelCase keys for frontend compatibility."""
        data = super().model_dump(**kwargs)
        return {
            "content": data["content"],
            "chunkType": data["chunk_type"],
            "taskId": data["task_id"],
            "taskName": data["task_name"],
            "taskIndex": data["task_index"],
            "agentRole": data["agent_role"],
            "agentId": data["agent_id"],
            "timestamp": data["timestamp"],
            "toolCall": data["tool_call"],
        }


class AgentInfo(BaseModel):
    """Agent metadata."""
    id: str
    role: str
    goal: str
    backstory: str
    crew: str  # staff, spot, futures
    status: str = "idle"  # idle, active, error


class CrewInfo(BaseModel):
    """Crew metadata."""
    id: str
    name: str
    description: str
    agent_count: int
    status: str = "stopped"  # stopped, running, error


class CrewStartRequest(BaseModel):
    """Request to start a crew."""
    inputs: dict[str, Any] = {}


class ChatMessage(BaseModel):
    """Chat message format."""
    role: str  # user, assistant, system
    content: str


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    crew_executed: bool = False
    crew_result: str | None = None


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections per crew."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.crew_tasks: dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, crew_id: str):
        await websocket.accept()
        if crew_id not in self.active_connections:
            self.active_connections[crew_id] = []
        self.active_connections[crew_id].append(websocket)

    def disconnect(self, websocket: WebSocket, crew_id: str):
        if crew_id in self.active_connections:
            if websocket in self.active_connections[crew_id]:
                self.active_connections[crew_id].remove(websocket)
            if not self.active_connections[crew_id]:
                del self.active_connections[crew_id]

    async def broadcast(self, crew_id: str, chunk: StreamChunk):
        """Send message to all clients connected to a crew."""
        if crew_id in self.active_connections:
            message = chunk.model_dump()
            disconnected = []
            for connection in self.active_connections[crew_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn, crew_id)

    async def send_heartbeat(self, crew_id: str):
        """Send heartbeat to keep connections alive."""
        chunk = StreamChunk(
            content="",
            chunk_type="heartbeat",
        )
        await self.broadcast(crew_id, chunk)


manager = ConnectionManager()


# =============================================================================
# Agent Registry
# =============================================================================

def get_all_agents() -> list[AgentInfo]:
    """Get all 74 agents with metadata."""
    agents = []

    try:
        from krakenagents.agents.staff import get_all_staff_agents
        from krakenagents.agents.spot import get_all_spot_agents
        from krakenagents.agents.futures import get_all_futures_agents

        # STAFF agents
        for i, agent in enumerate(get_all_staff_agents()):
            role_parts = agent.role.split(" — ")
            agents.append(AgentInfo(
                id=f"staff-{i:02d}",
                role=role_parts[0] if role_parts else agent.role,
                goal=agent.goal[:200] if agent.goal else "",
                backstory=agent.backstory[:200] if agent.backstory else "",
                crew="staff",
            ))

        # Spot agents
        for i, agent in enumerate(get_all_spot_agents()):
            role_parts = agent.role.split(" — ")
            agents.append(AgentInfo(
                id=f"spot-{i:02d}",
                role=role_parts[0] if role_parts else agent.role,
                goal=agent.goal[:200] if agent.goal else "",
                backstory=agent.backstory[:200] if agent.backstory else "",
                crew="spot",
            ))

        # Futures agents
        for i, agent in enumerate(get_all_futures_agents()):
            role_parts = agent.role.split(" — ")
            agents.append(AgentInfo(
                id=f"futures-{i:02d}",
                role=role_parts[0] if role_parts else agent.role,
                goal=agent.goal[:200] if agent.goal else "",
                backstory=agent.backstory[:200] if agent.backstory else "",
                crew="futures",
            ))

    except ImportError as e:
        print(f"Warning: Could not import agents: {e}")

    return agents


def get_crews() -> list[CrewInfo]:
    """Get all available crews."""
    return [
        CrewInfo(
            id="staff",
            name="STAFF",
            description="Group Executive Board - C-level leadership",
            agent_count=10,
        ),
        CrewInfo(
            id="spot",
            name="Spot Desk",
            description="Spot trading operations",
            agent_count=32,
        ),
        CrewInfo(
            id="futures",
            name="Futures Desk",
            description="Futures/derivatives trading operations",
            agent_count=32,
        ),
    ]


# =============================================================================
# Crew Execution with Streaming
# =============================================================================

def convert_crewai_chunk(crewai_chunk: Any) -> StreamChunk:
    """Convert CrewAI StreamChunk to server StreamChunk format."""
    # Get chunk type as string
    chunk_type = crewai_chunk.chunk_type.value if hasattr(crewai_chunk.chunk_type, 'value') else str(crewai_chunk.chunk_type)

    # Convert tool_call to dict if present (camelCase for frontend)
    tool_call_dict = None
    if crewai_chunk.tool_call:
        tool_call_dict = {
            "toolId": crewai_chunk.tool_call.tool_id or "",
            "toolName": crewai_chunk.tool_call.tool_name or "",
            "name": crewai_chunk.tool_call.tool_name or "",  # Also include 'name' for compatibility
            "arguments": crewai_chunk.tool_call.arguments or "",
            "index": crewai_chunk.tool_call.index or 0,
        }

    # Convert timestamp
    timestamp = crewai_chunk.timestamp.isoformat() + "Z" if crewai_chunk.timestamp else datetime.utcnow().isoformat() + "Z"

    # Get agent role - use a fallback if empty
    agent_role = crewai_chunk.agent_role or ""
    agent_id = crewai_chunk.agent_id or ""

    # Debug logging for TEXT chunks
    if chunk_type == "text" and crewai_chunk.content:
        print(f"[STREAM] {chunk_type}: agent_role='{agent_role}', content='{crewai_chunk.content[:50]}...'")

    return StreamChunk(
        content=crewai_chunk.content or "",
        chunk_type=chunk_type,
        task_id=crewai_chunk.task_id or "",
        task_name=crewai_chunk.task_name or "",
        task_index=crewai_chunk.task_index or 0,
        agent_role=agent_role,
        agent_id=agent_id,
        timestamp=timestamp,
        tool_call=tool_call_dict,
    )


async def run_crew_with_streaming(crew_id: str, inputs: dict[str, Any]):
    """Run a crew and stream output to connected clients."""
    from krakenagents.crews import (
        create_staff_crew,
        create_spot_crew,
        create_futures_crew,
    )

    try:
        # Create the appropriate crew
        if crew_id == "staff":
            crew = create_staff_crew()
        elif crew_id == "spot":
            crew = create_spot_crew()
        elif crew_id == "futures":
            crew = create_futures_crew()
        else:
            raise ValueError(f"Unknown crew: {crew_id}")

        loop = asyncio.get_event_loop()

        # Queue for passing chunks from thread to async
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()

        def run_kickoff_with_streaming():
            """Run crew in thread, put chunks in queue."""
            try:
                # crew.kickoff returns CrewStreamingOutput when stream=True
                streaming_output = crew.kickoff(inputs=inputs)

                # Iterate over all chunks and queue them
                for chunk in streaming_output:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", chunk))

                # Signal completion with final result
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", streaming_output.result))

            except Exception as e:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", e))

        # Start crew execution in background thread
        loop.run_in_executor(None, run_kickoff_with_streaming)

        # Process chunks as they arrive
        while True:
            item = await chunk_queue.get()
            msg_type, data = item

            if msg_type == "chunk":
                # Convert CrewAI chunk to server format and broadcast
                server_chunk = convert_crewai_chunk(data)
                await manager.broadcast(crew_id, server_chunk)

            elif msg_type == "done":
                # Send final completion message
                await manager.broadcast(crew_id, StreamChunk(
                    content=str(data) if data else "Crew execution completed",
                    chunk_type="task_completed",
                    agent_role=f"{crew_id.title()} Manager",
                    agent_id=f"{crew_id}-manager",
                ))
                break

            elif msg_type == "error":
                await manager.broadcast(crew_id, StreamChunk(
                    content=str(data),
                    chunk_type="error",
                    agent_role=f"{crew_id.title()} Manager",
                    agent_id=f"{crew_id}-manager",
                ))
                break

    except Exception as e:
        await manager.broadcast(crew_id, StreamChunk(
            content=str(e),
            chunk_type="error",
            agent_role=f"{crew_id.title()} Manager",
            agent_id=f"{crew_id}-manager",
        ))


# =============================================================================
# Chat Service
# =============================================================================

# Store chat sessions per crew
chat_sessions: dict[str, list[dict[str, str]]] = {}


def get_crew_for_chat(crew_id: str):
    """Get crew instance for chat."""
    from krakenagents.crews import (
        create_staff_crew,
        create_spot_crew,
        create_futures_crew,
    )

    if crew_id == "staff":
        return create_staff_crew()
    elif crew_id == "spot":
        return create_spot_crew()
    elif crew_id == "futures":
        return create_futures_crew()
    else:
        raise ValueError(f"Unknown crew: {crew_id}")


def build_crew_system_message(crew_id: str) -> str:
    """Build system message for crew chat."""
    crew_descriptions = {
        "staff": "Je bent de assistent voor het STAFF team (Group Executive Board). Je helpt met governance, risk oversight en strategische beslissingen. Je kunt de CEO en andere C-level executives aansturen voor taken.",
        "spot": "Je bent de assistent voor de Spot Trading Desk. Je helpt met spot trading operaties, marktanalyse, en uitvoering. Je kunt het team van 32 trading agents aansturen voor taken.",
        "futures": "Je bent de assistent voor de Futures Trading Desk. Je helpt met futures/derivatives trading, carry trades, en risk management. Je kunt het team van 32 futures agents aansturen voor taken.",
    }

    base_message = crew_descriptions.get(crew_id, "Je bent een trading assistent.")

    return f"""
{base_message}

Je kunt de volgende acties uitvoeren:
1. Beantwoord vragen over trading, marktanalyse, en risicobeheer
2. Start crew taken wanneer de gebruiker vraagt om analyses of acties
3. Geef informatie over de agents in je team

Als de gebruiker vraagt om een actie uit te voeren (bijv. "analyseer de markt", "maak een rapport"),
geef aan dat je de crew gaat inschakelen en wacht op de resultaten.

Houd je antwoorden beknopt en professioneel. Spreek in het Nederlands.
"""


async def process_chat_message(
    crew_id: str,
    message: str,
    history: list[dict[str, str]],
    websocket: WebSocket | None = None,
) -> str:
    """Process a chat message and return response."""
    from crewai.utilities.llm_utils import create_llm
    from krakenagents.config import get_chat_llm

    # Initialize session if needed
    if crew_id not in chat_sessions:
        chat_sessions[crew_id] = []

    # Get chat LLM
    chat_llm = get_chat_llm()

    # Build messages
    system_message = build_crew_system_message(crew_id)
    messages = [{"role": "system", "content": system_message}]

    # Add history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Call LLM
    try:
        response = chat_llm.call(messages=messages)

        # Store in session
        chat_sessions[crew_id].append({"role": "user", "content": message})
        chat_sessions[crew_id].append({"role": "assistant", "content": str(response)})

        # Broadcast to WebSocket if connected
        if websocket:
            await manager.broadcast(crew_id, StreamChunk(
                content=str(response),
                chunk_type="text",
                agent_role="Chat Assistant",
                agent_id="chat-assistant",
            ))

        return str(response)

    except Exception as e:
        error_msg = f"Fout bij verwerken van bericht: {str(e)}"
        if websocket:
            await manager.broadcast(crew_id, StreamChunk(
                content=error_msg,
                chunk_type="error",
                agent_role="Chat Assistant",
                agent_id="chat-assistant",
            ))
        return error_msg


# =============================================================================
# Kraken Data Fetching
# =============================================================================

def _get_spot_credentials() -> dict[str, str]:
    """Get Kraken Spot API credentials from settings."""
    from krakenagents.config import get_settings
    settings = get_settings()
    return {
        "api_key": settings.kraken_api_key,
        "api_secret": settings.kraken_api_secret,
    }


def _get_futures_credentials() -> dict[str, str]:
    """Get Kraken Futures API credentials from settings."""
    from krakenagents.config import get_settings
    settings = get_settings()
    return {
        "api_key": settings.kraken_futures_api_key,
        "api_secret": settings.kraken_futures_api_secret,
    }


async def get_kraken_balance() -> dict[str, Any]:
    """Get Kraken account balance."""
    try:
        from crewai.tools.kraken import GetAccountBalanceTool
        creds = _get_spot_credentials()
        tool = GetAccountBalanceTool(**creds)
        result = tool._run()
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_kraken_positions() -> dict[str, Any]:
    """Get Kraken open positions."""
    try:
        from crewai.tools.kraken import GetOpenPositionsTool
        creds = _get_spot_credentials()
        tool = GetOpenPositionsTool(**creds)
        result = tool._run()
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_kraken_orders() -> dict[str, Any]:
    """Get Kraken open orders."""
    try:
        from crewai.tools.kraken import GetOpenOrdersTool
        creds = _get_spot_credentials()
        tool = GetOpenOrdersTool(**creds)
        result = tool._run()
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_kraken_ticker(pair: str = "XBTUSD") -> dict[str, Any]:
    """Get Kraken ticker for a pair."""
    try:
        from crewai.tools.kraken import GetTickerInformationTool
        tool = GetTickerInformationTool()  # Public endpoint, no auth needed
        result = tool._run(pair=pair)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_kraken_futures_wallets() -> dict[str, Any]:
    """Get Kraken Futures wallets."""
    try:
        from crewai.tools.kraken.futures import KrakenFuturesGetWalletsTool
        creds = _get_futures_credentials()
        tool = KrakenFuturesGetWalletsTool(**creds)
        result = tool._run()
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("QRI Trading Server starting...")

    # Initialize crew manager for delegation tools
    try:
        from krakenagents.tools.crew_delegation import set_crew_manager
        loop = asyncio.get_event_loop()
        set_crew_manager(manager, loop)
        print("Crew delegation tools initialized - STAFF can now control Spot/Futures desks")
    except ImportError as e:
        print(f"Warning: Could not initialize delegation tools: {e}")

    yield
    print("QRI Trading Server shutting down...")
    # Cancel any running crew tasks
    for task in manager.crew_tasks.values():
        task.cancel()


app = FastAPI(
    title="QRI Trading API",
    description="WebSocket server for real-time agent streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include management API router
app.include_router(management_router)


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "qri-trading",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/agents", response_model=list[AgentInfo])
async def list_agents():
    """Get all agents."""
    return get_all_agents()


@app.get("/api/crews", response_model=list[CrewInfo])
async def list_crews():
    """Get all crews."""
    return get_crews()


@app.post("/api/crews/{crew_id}/start")
async def start_crew(crew_id: str, request: CrewStartRequest):
    """Start a crew execution."""
    if crew_id not in ["staff", "spot", "futures"]:
        raise HTTPException(status_code=404, detail=f"Crew {crew_id} not found")

    if crew_id in manager.crew_tasks and not manager.crew_tasks[crew_id].done():
        raise HTTPException(status_code=400, detail=f"Crew {crew_id} is already running")

    # Start crew execution in background
    task = asyncio.create_task(run_crew_with_streaming(crew_id, request.inputs))
    manager.crew_tasks[crew_id] = task

    return {"status": "started", "crew_id": crew_id}


@app.post("/api/crews/{crew_id}/stop")
async def stop_crew(crew_id: str):
    """Stop a crew execution."""
    if crew_id in manager.crew_tasks:
        manager.crew_tasks[crew_id].cancel()
        del manager.crew_tasks[crew_id]
        return {"status": "stopped", "crew_id": crew_id}
    return {"status": "not_running", "crew_id": crew_id}


# Kraken data endpoints
@app.get("/api/kraken/balance")
async def kraken_balance():
    """Get Kraken account balance."""
    return await get_kraken_balance()


@app.get("/api/kraken/positions")
async def kraken_positions():
    """Get Kraken open positions."""
    return await get_kraken_positions()


@app.get("/api/kraken/orders")
async def kraken_orders():
    """Get Kraken open orders."""
    return await get_kraken_orders()


@app.get("/api/kraken/ticker/{pair}")
async def kraken_ticker(pair: str = "XBTUSD"):
    """Get Kraken ticker for a pair."""
    return await get_kraken_ticker(pair)


@app.get("/api/kraken/futures/wallets")
async def kraken_futures_wallets():
    """Get Kraken Futures wallets."""
    return await get_kraken_futures_wallets()


# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post("/api/chat/{crew_id}", response_model=ChatResponse)
async def chat_with_crew(crew_id: str, request: ChatRequest):
    """Send a chat message to a crew."""
    if crew_id not in ["staff", "spot", "futures"]:
        raise HTTPException(status_code=404, detail=f"Crew {crew_id} not found")

    # Convert history to dict format
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # Process message
    response = await process_chat_message(crew_id, request.message, history)

    return ChatResponse(
        response=response,
        crew_executed=False,
        crew_result=None,
    )


@app.post("/api/chat/{crew_id}/clear")
async def clear_chat_session(crew_id: str):
    """Clear chat session for a crew."""
    if crew_id in chat_sessions:
        chat_sessions[crew_id] = []
    return {"status": "cleared", "crew_id": crew_id}


@app.get("/api/chat/{crew_id}/history")
async def get_chat_history(crew_id: str):
    """Get chat history for a crew."""
    return {"history": chat_sessions.get(crew_id, [])}


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws/{crew_id}")
async def websocket_endpoint(websocket: WebSocket, crew_id: str):
    """WebSocket endpoint for real-time crew updates and chat."""
    await manager.connect(websocket, crew_id)

    # Send connected confirmation (as info type, not text - so it won't create a chat message)
    await websocket.send_json({
        "content": f"Connected to {crew_id} crew stream",
        "chunkType": "info",
        "agentRole": "System",
        "agentId": "system",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    try:
        while True:
            # Wait for messages from client (commands or chat)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Handle client commands
                if message.get("command") == "start":
                    inputs = message.get("inputs", {})
                    if crew_id not in manager.crew_tasks or manager.crew_tasks[crew_id].done():
                        task = asyncio.create_task(run_crew_with_streaming(crew_id, inputs))
                        manager.crew_tasks[crew_id] = task
                        await websocket.send_json({"status": "started"})
                    else:
                        await websocket.send_json({"status": "already_running"})

                elif message.get("command") == "stop":
                    if crew_id in manager.crew_tasks:
                        manager.crew_tasks[crew_id].cancel()
                        await websocket.send_json({"status": "stopped"})

                # Handle chat messages
                elif message.get("type") == "chat":
                    chat_content = message.get("content", "")
                    chat_history = message.get("history", [])

                    # Send typing indicator
                    await websocket.send_json({
                        "chunk_type": "chat_typing",
                        "content": "",
                        "agent_role": "Chat Assistant",
                        "agent_id": "chat-assistant",
                    })

                    # Process chat message
                    response = await process_chat_message(
                        crew_id,
                        chat_content,
                        chat_history,
                        websocket,
                    )

                    # Send chat response
                    await websocket.send_json({
                        "chunk_type": "chat_response",
                        "content": response,
                        "agent_role": "Chat Assistant",
                        "agent_id": "chat-assistant",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await manager.send_heartbeat(crew_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, crew_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, crew_id)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

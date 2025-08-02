import asyncio
import json
import queue
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
    from fastapi.responses import StreamingResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .event_stream_manager import event_stream_manager


class HumanInputServer:
    """HTTP server for human input event streaming"""

    def __init__(self, host: str = "localhost", port: int = 8000, api_key: Optional[str] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI dependencies not available. Install with: pip install fastapi uvicorn websockets"
            )
        
        self.host = host
        self.port = port
        self.api_key = api_key
        self.app = FastAPI(title="CrewAI Human Input Event Stream API")
        self.security = HTTPBearer() if api_key else None
        self._setup_routes()
        event_stream_manager.register_event_listeners()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.websocket("/ws/human-input/{execution_id}")
        async def websocket_endpoint(websocket: WebSocket, execution_id: str):
            if self.api_key:
                token = websocket.query_params.get("token")
                if not token or token != self.api_key:
                    await websocket.close(code=4001, reason="Unauthorized")
                    return

            await websocket.accept()
            event_stream_manager.add_websocket_connection(execution_id, websocket)
            
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                event_stream_manager.remove_websocket_connection(execution_id, websocket)

        @self.app.get("/events/human-input/{execution_id}")
        async def sse_endpoint(execution_id: str, credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security) if self.security else None):
            if self.api_key and (not credentials or credentials.credentials != self.api_key):
                raise HTTPException(status_code=401, detail="Unauthorized")

            async def event_stream():
                event_queue = asyncio.Queue()
                event_stream_manager.add_sse_connection(execution_id, event_queue)
                
                try:
                    while True:
                        try:
                            event_data = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                            yield f"data: {json.dumps(event_data)}\n\n"
                        except asyncio.TimeoutError:
                            yield "data: {\"type\": \"heartbeat\"}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    event_stream_manager.remove_sse_connection(execution_id, event_queue)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )

        @self.app.get("/poll/human-input/{execution_id}")
        async def polling_endpoint(
            execution_id: str,
            last_event_id: Optional[str] = Query(None),
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security) if self.security else None
        ):
            if self.api_key and (not credentials or credentials.credentials != self.api_key):
                raise HTTPException(status_code=401, detail="Unauthorized")

            events = event_stream_manager.get_polling_events(execution_id, last_event_id)
            return {"events": events}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

    async def start_async(self):
        """Start the server asynchronously"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def start(self):
        """Start the server synchronously"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

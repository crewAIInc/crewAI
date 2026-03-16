"""Serves a CrewAI Crew via BedrockAgentCoreApp (POST /invocations, GET /ping)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.types import Lifespan

from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext


if TYPE_CHECKING:
    from crewai.crew import Crew

logger = logging.getLogger(__name__)


class AgentCoreRuntime:
    """Serves a CrewAI Crew via BedrockAgentCoreApp (POST /invocations, GET /ping).

    Example::

        from crewai import Crew, Agent, Task
        from crewai_tools.aws.bedrock.runtime import AgentCoreRuntime

        crew = Crew(agents=[...], tasks=[...])

        # One-liner
        AgentCoreRuntime.serve(crew)

        # Or with options
        runtime = AgentCoreRuntime(crew, stream=True, port=8080)
        runtime.run()
    """

    def __init__(
        self,
        crew: Crew,
        stream: bool = True,
        port: int = 8080,
        host: Optional[str] = None,
        debug: bool = False,
        lifespan: Optional[Lifespan] = None,
        middleware: Optional[Sequence[Middleware]] = None,
    ):
        self._crew = crew
        self._stream = stream
        self._port = port
        self._host = host
        self._app = BedrockAgentCoreApp(
            debug=debug, lifespan=lifespan, middleware=middleware
        )

        # Register entrypoint using closure wrappers (not bound methods).
        # entrypoint() sets func.run attr which fails on bound methods.
        # Closures also preserve isasyncgenfunction() detection for streaming.
        runtime = self
        if stream:

            async def streaming_entrypoint(
                payload: dict, context: RequestContext
            ) -> AsyncGenerator[dict, None]:
                async for chunk in runtime._streaming_handler(payload, context):
                    yield chunk

            self._app.entrypoint(streaming_entrypoint)
        else:

            async def non_streaming_entrypoint(
                payload: dict, context: RequestContext
            ) -> dict:
                return await runtime._non_streaming_handler(payload, context)

            self._app.entrypoint(non_streaming_entrypoint)

    @classmethod
    def serve(cls, crew: Crew, **kwargs: Any) -> None:
        """Create runtime and start server in one call."""
        runtime = cls(crew=crew, **kwargs)
        runtime.run()

    def run(self, **kwargs: Any) -> None:
        """Start uvicorn server."""
        self._app.run(port=self._port, host=self._host, **kwargs)

    @property
    def app(self) -> BedrockAgentCoreApp:
        """Expose for ASGI mounting or testing."""
        return self._app

    @staticmethod
    def _extract_inputs(payload: dict) -> dict[str, Any]:
        """Normalize payload to CrewAI inputs dict.

        Accepts:
        - ``{"inputs": {"topic": "AI"}}`` — standard CrewAI inputs
        - ``{"prompt": "hello"}`` / ``{"message": "hello"}`` / ``{"input": "hello"}``
          — wrapped as ``{"input": prompt_str}``

        Raises HTTPException(400) if no usable input found.
        """
        # Standard CrewAI inputs dict
        inputs = payload.get("inputs")
        if isinstance(inputs, dict):
            return inputs

        # Fall back to prompt-style input (wrap as {"input": str})
        prompt = (
            payload.get("prompt") or payload.get("message") or payload.get("input")
        )
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt")
        if isinstance(prompt, str):
            prompt = prompt.strip()
        if prompt and isinstance(prompt, (str, int, float)):
            return {"input": str(prompt)}

        raise HTTPException(
            status_code=400,
            detail="Request must include 'inputs' (dict), 'prompt', 'message',"
            " or 'input' field",
        )

    async def _non_streaming_handler(
        self, payload: dict, context: RequestContext
    ) -> dict:
        """Handle non-streaming invocation. Returns JSON response."""
        inputs = self._extract_inputs(payload)
        crew = self._crew.copy()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: crew.kickoff(inputs=inputs)
        )
        return self._crew_output_to_dict(result)

    async def _streaming_handler(
        self, payload: dict, context: RequestContext
    ) -> AsyncGenerator[dict, None]:
        """Handle streaming invocation. Yields SSE event dicts."""
        inputs = self._extract_inputs(payload)

        # Create a per-request copy with streaming enabled to avoid mutating
        # the shared self._crew (which would be a race under concurrent requests).
        crew = self._crew.copy()
        crew.stream = True

        try:
            loop = asyncio.get_event_loop()
            streaming_output = await loop.run_in_executor(
                None, lambda: crew.kickoff(inputs=inputs)
            )

            # CrewStreamingOutput has a sync iterator — drain it on the executor
            # and push chunks to an async queue for the SSE generator.
            queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

            async def drain_chunks() -> None:
                try:
                    def _iter():
                        for chunk in streaming_output:
                            ev = self._stream_chunk_to_dict(chunk)
                            asyncio.run_coroutine_threadsafe(
                                queue.put(ev), loop
                            ).result()
                    await loop.run_in_executor(None, _iter)

                    # Send final "done" event with full result
                    try:
                        crew_result = streaming_output.result
                        done_ev = self._crew_output_to_dict(crew_result)
                        done_ev["event"] = "done"
                        await queue.put(done_ev)
                    except RuntimeError:
                        await queue.put({"event": "done", "response": streaming_output.get_full_text()})
                except Exception as e:
                    logger.exception("Error during streaming")
                    await queue.put({"event": "error", "message": str(e)})
                finally:
                    await queue.put(None)  # Sentinel

            drain_task = asyncio.create_task(drain_chunks())

            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield ev

            await drain_task

        except Exception as e:
            logger.exception("Error setting up streaming")
            yield {"event": "error", "message": str(e)}

    @staticmethod
    def _stream_chunk_to_dict(chunk: Any) -> Dict[str, Any]:
        """Convert a CrewAI StreamChunk to an SSE-compatible dict."""
        from crewai.types.streaming import StreamChunkType

        if chunk.chunk_type == StreamChunkType.TOOL_CALL and chunk.tool_call:
            return {
                "event": "tool_call",
                "tool_name": chunk.tool_call.tool_name or "",
                "arguments": chunk.tool_call.arguments,
                "agent_role": chunk.agent_role,
                "task_name": chunk.task_name,
            }
        return {
            "event": "text",
            "content": chunk.content,
            "agent_role": chunk.agent_role,
            "task_name": chunk.task_name,
        }

    @staticmethod
    def _crew_output_to_dict(result: Any) -> dict:
        """Convert a CrewOutput to a JSON-serializable response dict."""
        response: dict[str, Any] = {"response": str(result)}

        if hasattr(result, "json_dict") and result.json_dict:
            response["json"] = result.json_dict

        if hasattr(result, "tasks_output") and result.tasks_output:
            response["tasks_output"] = [
                {
                    "agent": getattr(t, "agent", ""),
                    "description": getattr(t, "description", ""),
                    "raw": getattr(t, "raw", ""),
                }
                for t in result.tasks_output
            ]

        if hasattr(result, "token_usage") and result.token_usage:
            usage = result.token_usage
            response["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }

        return response

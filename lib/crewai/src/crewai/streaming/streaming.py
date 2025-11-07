import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from crewai.events import (
    AgentExecutionCompletedEvent, 
    BaseEventListener,
    CrewKickoffCompletedEvent, 
    LLMCallCompletedEvent,
    LLMStreamChunkEvent, 
    TaskCompletedEvent
)

# Configure logger
logger = logging.getLogger(__name__)


class CrewStreamListener(BaseEventListener):
    """Event listener for streaming CrewAI execution tokens."""
    
    END_OF_STREAM = "END_OF_STREAM"
    
    def __init__(self, target_agent_ids: List[str], crew_id: Optional[str] = None):
        """
        Initialize the stream listener.
        
        Args:
            target_agent_ids: List of agent IDs to monitor for streaming
            crew_id: Optional crew ID to monitor for completion
        """
        super().__init__()
        self.target_agent_ids = [str(agent_id) for agent_id in target_agent_ids]
        self.crew_id = crew_id
        self.event_queue = asyncio.Queue()
        self._is_streaming = False
        
    def setup_listeners(self, event_bus):
        """Setup event listeners on the CrewAI event bus."""
        
        @event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source: Any, event: LLMStreamChunkEvent):
            """Handle LLM stream chunk events."""
            if str(event.agent_id) in self.target_agent_ids:
                logger.debug(f"Received stream chunk from agent {event.agent_id}")
                self.event_queue.put_nowait(event.chunk)
                
        @event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_kickoff_complete(source: Any, event: CrewKickoffCompletedEvent):
            """Handle crew completion events."""
            if self.crew_id and source.fingerprint.uuid_str == self.crew_id:
                logger.info(f"Crew {self.crew_id} completed")
                self.event_queue.put_nowait(self.END_OF_STREAM)
            elif not self.crew_id:
                # If no specific crew_id, end stream on any crew completion
                logger.info("Crew execution completed")
                self.event_queue.put_nowait(self.END_OF_STREAM)
                
    async def get_tokens(self) -> AsyncGenerator[str, None]:
        """
        Get streaming tokens as they arrive.
        
        Yields:
            str: Stream tokens from the target agents
        """
        self._is_streaming = True
        try:
            while self._is_streaming:
                token = await self.event_queue.get()
                
                if token == self.END_OF_STREAM:
                    logger.info("Stream ended")
                    break
                    
                yield token
                self.event_queue.task_done()
        except Exception as e:
            logger.error(f"Error in token streaming: {e}")
            raise
        finally:
            self._is_streaming = False
            
    def stop_streaming(self):
        """Stop the streaming manually."""
        self._is_streaming = False
        self.event_queue.put_nowait(self.END_OF_STREAM)


class CrewStreamer:
    """
    High-level interface for streaming CrewAI execution.
    
    This class provides a simple way to stream tokens from CrewAI agents
    without needing to understand the underlying event system.
    """
    
    def __init__(self, crew_instance, agent_ids: Optional[List[str]] = None):
        """
        Initialize the CrewStreamer.
        
        Args:
            crew_instance: The CrewAI crew instance to stream from
            agent_ids: Optional list of specific agent IDs to monitor.
                      If None, will monitor all agents in the crew.
        """
        self.crew_instance = crew_instance
        self.crew = crew_instance.crew()
        
        # If no agent_ids specified, get all agent IDs from the crew
        if agent_ids is None:
            self.agent_ids = [str(agent.id) for agent in self.crew.agents]
        else:
            self.agent_ids = [str(agent_id) for agent_id in agent_ids]
            
        self.crew_id = self.crew.fingerprint.uuid_str
        self.listener = None
        
    async def stream_execution(
        self, 
        inputs: Dict[str, Any], 
        sleep_time: float = 0.01,
        wait_for_final_answer: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Stream the crew execution with real-time tokens.
        
        Args:
            inputs: Input data for the crew execution
            sleep_time: Sleep time between token yields (for rate limiting)
            wait_for_final_answer: If True, only start yielding tokens after "Final Answer:" appears
            
        Yields:
            str: Streaming tokens from the crew execution
            
        Raises:
            Exception: Any errors during crew execution or streaming
        """
        try:
            # Create and setup the listener
            self.listener = CrewStreamListener(
                target_agent_ids=self.agent_ids,
                crew_id=self.crew_id
            )
            
            # Start the crew execution task
            execution_task = asyncio.create_task(
                self.crew.kickoff_async(inputs=inputs)
            )
            
            # Stream tokens
            accumulated_result = ""
            final_answer_reached = not wait_for_final_answer
            
            async for token in self.listener.get_tokens():
                accumulated_result += token
                
                # Check if we should start yielding tokens
                if not final_answer_reached and "Final Answer:" in accumulated_result:
                    final_answer_reached = True
                    logger.info("Final Answer section reached, starting token stream")
                
                if final_answer_reached:
                    yield token
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Ensure the execution task completes
            if not execution_task.done():
                logger.info("Waiting for crew execution to complete...")
                await execution_task
                
        except asyncio.CancelledError:
            logger.info("Streaming was cancelled")
            if hasattr(self, 'listener') and self.listener:
                self.listener.stop_streaming()
            raise
        except Exception as e:
            logger.error(f"Error during crew streaming: {e}")
            if hasattr(self, 'listener') and self.listener:
                self.listener.stop_streaming()
            raise
            
    def stop(self):
        """Stop the streaming manually."""
        if self.listener:
            self.listener.stop_streaming()


# Convenience function for backward compatibility and simple usage
async def stream_crew_execution(
    crew_instance, 
    inputs: Dict[str, Any],
    agent_ids: Optional[List[str]] = None,
    sleep_time: float = 0.01,
    wait_for_final_answer: bool = True
) -> AsyncGenerator[str, None]:
    """
    Convenience function to stream crew execution.
    
    Args:
        crew_instance: The CrewAI crew instance
        inputs: Input data for the crew execution
        agent_ids: Optional list of agent IDs to monitor
        sleep_time: Sleep time between tokens
        wait_for_final_answer: Whether to wait for "Final Answer:" before streaming
        
    Yields:
        str: Streaming tokens
        
    Example:
        ```python
        async for token in stream_crew_execution(my_crew, {"input": "Hello"}):
            print(token, end="", flush=True)
        ```
    """
    streamer = CrewStreamer(crew_instance, agent_ids)
    async for token in streamer.stream_execution(inputs, sleep_time, wait_for_final_answer):
        yield token
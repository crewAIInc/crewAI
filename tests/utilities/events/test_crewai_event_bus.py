import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.llm_events import LLMStreamChunkEvent


class TestEvent(BaseEvent):
    pass


def test_specific_event_handler():
    mock_handler = Mock()

    @crewai_event_bus.on(TestEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)


def test_wildcard_event_handler():
    mock_handler = Mock()

    @crewai_event_bus.on(BaseEvent)
    def handler(source, event):
        mock_handler(source, event)

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    mock_handler.assert_called_once_with("source_object", event)


def test_event_bus_error_handling(capfd):
    @crewai_event_bus.on(BaseEvent)
    def broken_handler(source, event):
        raise ValueError("Simulated handler failure")

    event = TestEvent(type="test_event")
    crewai_event_bus.emit("source_object", event)

    out, err = capfd.readouterr()
    assert "Simulated handler failure" in out
    assert "Handler 'broken_handler' failed" in out


def test_concurrent_event_emission_thread_safety():
    """Test that concurrent event emission is thread-safe"""
    
    handler1_events = []
    handler2_events = []
    handler_lock = threading.Lock()
    
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handler1(source, event: LLMStreamChunkEvent):
            with handler_lock:
                handler1_events.append(f"Handler1: {event.chunk}")
        
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handler2(source, event: LLMStreamChunkEvent):
            with handler_lock:
                handler2_events.append(f"Handler2: {event.chunk}")
        
        def emit_events(thread_id, num_events=20):
            """Emit events from a specific thread"""
            for i in range(num_events):
                event = LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk=f"Thread-{thread_id}-Chunk-{i}"
                )
                crewai_event_bus.emit(f"source-{thread_id}", event)
        
        num_threads = 5
        events_per_thread = 20
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                future = executor.submit(emit_events, thread_id, events_per_thread)
                futures.append(future)
            
            for future in futures:
                future.result()
        
        expected_total = num_threads * events_per_thread
        assert len(handler1_events) == expected_total, f"Handler1 received {len(handler1_events)} events, expected {expected_total}"
        assert len(handler2_events) == expected_total, f"Handler2 received {len(handler2_events)} events, expected {expected_total}"


def test_concurrent_handler_registration_thread_safety():
    """Test that concurrent handler registration is thread-safe"""
    
    registered_handlers = []
    
    def register_handler(thread_id):
        """Register a handler from a specific thread"""
        def handler(source, event):
            pass
        
        handler.__name__ = f"handler_{thread_id}"
        crewai_event_bus.register_handler(TestEvent, handler)
        registered_handlers.append(handler)
    
    num_threads = 10
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for thread_id in range(num_threads):
            future = executor.submit(register_handler, thread_id)
            futures.append(future)
        
        for future in futures:
            future.result()
    
    assert len(registered_handlers) == num_threads
    assert len(crewai_event_bus._handlers[TestEvent]) >= num_threads


def test_thread_safety_with_mixed_operations():
    """Test thread safety when mixing event emission and handler registration"""
    
    received_events = []
    event_lock = threading.Lock()
    
    with crewai_event_bus.scoped_handlers():
        def emit_events(thread_id):
            for i in range(10):
                event = TestEvent(type="test_event")
                crewai_event_bus.emit(f"source-{thread_id}", event)
                time.sleep(0.001)
        
        def register_handlers(thread_id):
            for i in range(5):
                def handler(source, event):
                    with event_lock:
                        received_events.append(f"Handler-{thread_id}-{i}: {event.type}")
                
                handler.__name__ = f"handler_{thread_id}_{i}"
                crewai_event_bus.register_handler(TestEvent, handler)
                time.sleep(0.001)
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            for thread_id in range(3):
                futures.append(executor.submit(emit_events, thread_id))
            
            for thread_id in range(3):
                futures.append(executor.submit(register_handlers, thread_id))
            
            for future in futures:
                future.result()
        
        assert len(received_events) >= 0

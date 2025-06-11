#!/usr/bin/env python3
"""
Simple verification script for thread safety fix without pytest dependencies.
"""

import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic event emission works"""
    print("Testing basic functionality...")
    
    from crewai.utilities.events.crewai_event_bus import crewai_event_bus
    from crewai.utilities.events.llm_events import LLMStreamChunkEvent
    
    received_events = []
    
    @crewai_event_bus.on(LLMStreamChunkEvent)
    def handler(source, event):
        received_events.append(f"{source}: {event.chunk}")
    
    event = LLMStreamChunkEvent(type='llm_stream_chunk', chunk='test')
    crewai_event_bus.emit('test_source', event)
    
    if len(received_events) == 1 and 'test_source: test' in received_events[0]:
        print("✅ Basic event emission works")
        return True
    else:
        print("❌ Basic event emission failed")
        print(f"Received: {received_events}")
        return False

def test_thread_safety():
    """Test thread safety of concurrent event emission"""
    print("Testing thread safety...")
    
    from crewai.utilities.events.crewai_event_bus import crewai_event_bus
    from crewai.utilities.events.llm_events import LLMStreamChunkEvent
    
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
        
        def emit_events(thread_id, num_events=10):
            """Emit events from a specific thread"""
            for i in range(num_events):
                event = LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk=f"Thread-{thread_id}-Chunk-{i}"
                )
                crewai_event_bus.emit(f"source-{thread_id}", event)
        
        num_threads = 3
        events_per_thread = 10
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                future = executor.submit(emit_events, thread_id, events_per_thread)
                futures.append(future)
            
            for future in futures:
                future.result()
        
        expected_total = num_threads * events_per_thread
        success = (len(handler1_events) == expected_total and 
                  len(handler2_events) == expected_total)
        
        if success:
            print(f"✅ Thread safety test passed - each handler received {expected_total} events")
            return True
        else:
            print(f"❌ Thread safety test failed")
            print(f"Handler1 received {len(handler1_events)} events, expected {expected_total}")
            print(f"Handler2 received {len(handler2_events)} events, expected {expected_total}")
            return False

def test_deregistration():
    """Test handler deregistration"""
    print("Testing handler deregistration...")
    
    from crewai.utilities.events.crewai_event_bus import crewai_event_bus
    from crewai.utilities.events.base_events import BaseEvent
    
    class TestEvent(BaseEvent):
        pass
    
    with crewai_event_bus.scoped_handlers():
        def test_handler(source, event):
            pass
        
        crewai_event_bus.register_handler(TestEvent, test_handler)
        initial_count = len(crewai_event_bus._handlers.get(TestEvent, []))
        
        result = crewai_event_bus.deregister_handler(TestEvent, test_handler)
        final_count = len(crewai_event_bus._handlers.get(TestEvent, []))
        
        if result and final_count == 0 and initial_count == 1:
            print("✅ Handler deregistration works")
            return True
        else:
            print("❌ Handler deregistration failed")
            print(f"Initial count: {initial_count}, Final count: {final_count}, Result: {result}")
            return False

def main():
    print("Verifying thread safety fix for Issue #2991")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_thread_safety,
        test_deregistration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All thread safety tests passed!")
        print("The fix for Issue #2991 is working correctly.")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

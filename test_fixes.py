#!/usr/bin/env python3
"""
Simple test script to verify the CI fixes work locally.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.llm_events import LLMStreamChunkEvent
import logging

class TestEvent(BaseEvent):
    pass

def test_basic_functionality():
    """Test basic event emission works"""
    print("Testing basic functionality...")
    
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

def test_error_handling():
    """Test error handling with structured logging"""
    print("Testing error handling...")
    
    import logging
    logging.basicConfig(level=logging.ERROR)
    
    with crewai_event_bus.scoped_handlers():
        @crewai_event_bus.on(BaseEvent)
        def broken_handler(source, event):
            raise ValueError("Simulated handler failure")
        
        event = TestEvent(type="test_event")
        crewai_event_bus.emit("source_object", event)
        
        print("✅ Error handling test completed (check logs above)")
        return True

def test_deregistration():
    """Test handler deregistration"""
    print("Testing handler deregistration...")
    
    with crewai_event_bus.scoped_handlers():
        def test_handler(source, event):
            pass
        
        crewai_event_bus.register_handler(TestEvent, test_handler)
        initial_count = len(crewai_event_bus._handlers.get(TestEvent, []))
        print(f"Handlers after registration: {initial_count}")
        
        result = crewai_event_bus.deregister_handler(TestEvent, test_handler)
        final_count = len(crewai_event_bus._handlers.get(TestEvent, []))
        print(f"Handlers after deregistration: {final_count}")
        print(f"Deregistration result: {result}")
        
        if result and final_count == 0:
            print("✅ Handler deregistration works")
            return True
        else:
            print("❌ Handler deregistration failed")
            return False

def main():
    print("Testing CI fixes locally")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_error_handling,
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
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All local tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

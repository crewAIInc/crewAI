import threading

import pytest

from crewai.telemetry import Telemetry


def test_telemetry_singleton():
    """Test that Telemetry is a singleton and only one instance is created."""
    # Create multiple instances of Telemetry
    telemetry1 = Telemetry()
    telemetry2 = Telemetry()
    
    # Verify that they are the same instance
    assert telemetry1 is telemetry2
    
    # Verify that the BatchSpanProcessor is initialized only once
    # by checking that the provider is the same
    assert telemetry1.provider is telemetry2.provider


def test_telemetry_thread_safety():
    """Test that Telemetry singleton is thread-safe."""
    # List to store Telemetry instances created in threads
    instances = []
    
    def create_telemetry():
        instances.append(Telemetry())
    
    # Create multiple threads that create Telemetry instances
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=create_telemetry)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify that all instances are the same
    for instance in instances[1:]:
        assert instance is instances[0]

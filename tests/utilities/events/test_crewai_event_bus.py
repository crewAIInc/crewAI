import threading
from unittest.mock import Mock
import pytest

from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


@pytest.fixture(autouse=True)
def scoped_event_handlers():
    with crewai_event_bus.scoped_handlers():
        yield


class TestEvent(BaseEvent):
    pass


class AnotherThreadTestEvent(BaseEvent):
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


def test_singleton_pattern_across_threads():
    instances = []

    def get_instance():
        instances.append(crewai_event_bus)

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=get_instance)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    assert len(instances) == 10
    for instance in instances:
        assert instance is crewai_event_bus
        assert instance is instances[0]


def test_thread_local_handler_isolation():
    thread_results = {}

    def thread_worker(thread_id):
        mock_handler = Mock()

        @crewai_event_bus.on(TestEvent)
        def thread_handler(source, event):
            mock_handler(f"thread_{thread_id}", event)

        event = TestEvent(type=f"test_event_thread_{thread_id}")
        crewai_event_bus.emit(f"source_{thread_id}", event)

        thread_results[thread_id] = {
            'mock_handler': mock_handler,
            'handler_function': thread_handler,
            'event': event
        }

    threads = []
    for i in range(5):
        thread = threading.Thread(target=thread_worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert len(thread_results) == 5

    for thread_id, result in thread_results.items():
        result['mock_handler'].assert_called_once_with(
            f"thread_{thread_id}",
            result['event']
        )


def test_scoped_handlers_thread_safety():
    thread_results = {}

    def thread_worker(thread_id):
        with crewai_event_bus.scoped_handlers():
            mock_handler = Mock()

            @crewai_event_bus.on(AnotherThreadTestEvent)
            def scoped_handler(source, event):
                mock_handler(f"scoped_thread_{thread_id}", event)

            scoped_event = AnotherThreadTestEvent(type=f"scoped_event_{thread_id}")
            crewai_event_bus.emit(f"scoped_source_{thread_id}", scoped_event)

            thread_results[thread_id] = {
                'mock_handler': mock_handler,
                'scoped_event': scoped_event
            }

        post_scoped_event = AnotherThreadTestEvent(type=f"post_scoped_{thread_id}")
        crewai_event_bus.emit(f"post_source_{thread_id}", post_scoped_event)

    threads = []
    for i in range(5):
        thread = threading.Thread(target=thread_worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for thread_id, result in thread_results.items():
        result['mock_handler'].assert_called_once_with(
            f"scoped_thread_{thread_id}",
            result['scoped_event']
        )
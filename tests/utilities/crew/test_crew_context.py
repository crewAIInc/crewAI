import uuid

import pytest

from crewai.utilities.crew.crew_context import (
    CrewContext,
    crew_context,
    get_crew_context,
    with_crew_context,
)


def test_crew_context_creation():
    crew_id = uuid.uuid4()
    context = CrewContext(id=crew_id, key="test-crew")
    assert context.id == crew_id
    assert context.key == "test-crew"


def test_crew_context_manager():
    crew_id = uuid.uuid4()
    assert get_crew_context() is None

    with crew_context(crew_id=crew_id, crew_key="test-key"):
        context = get_crew_context()
        assert context is not None
        assert context.id == crew_id
        assert context.key == "test-key"

    assert get_crew_context() is None


def test_crew_context_manager_empty():
    assert get_crew_context() is None

    with crew_context():
        assert get_crew_context() is None

    assert get_crew_context() is None


def test_crew_context_nested():
    crew_id1 = uuid.uuid4()
    crew_id2 = uuid.uuid4()

    with crew_context(crew_id=crew_id1, crew_key="outer"):
        outer_context = get_crew_context()
        assert outer_context.id == crew_id1
        assert outer_context.key == "outer"

        with crew_context(crew_id=crew_id2, crew_key="inner"):
            inner_context = get_crew_context()
            assert inner_context.id == crew_id2
            assert inner_context.key == "inner"

        restored_context = get_crew_context()
        assert restored_context.id == crew_id1
        assert restored_context.key == "outer"

    assert get_crew_context() is None


def test_with_crew_context_decorator():
    class MockCrew:
        def __init__(self, id, key):
            self.id = id
            self.key = key

        @with_crew_context
        def execute(self):
            return get_crew_context()

    crew_id = uuid.uuid4()
    crew = MockCrew(id=crew_id, key="test-crew")

    result = crew.execute()
    assert result is not None
    assert result.id == crew_id
    assert result.key == "test-crew"

    assert get_crew_context() is None


def test_with_crew_context_decorator_no_attributes():
    class MockObject:
        @with_crew_context
        def execute(self):
            return get_crew_context()

    obj = MockObject()
    result = obj.execute()
    assert result is None


def test_with_crew_context_decorator_partial_attributes():
    class MockCrewOnlyId:
        def __init__(self, id):
            self.id = id

        @with_crew_context
        def execute(self):
            return get_crew_context()

    class MockCrewOnlyKey:
        def __init__(self, key):
            self.key = key

        @with_crew_context
        def execute(self):
            return get_crew_context()

    crew_id = uuid.uuid4()
    crew1 = MockCrewOnlyId(id=crew_id)
    result1 = crew1.execute()
    assert result1 is not None
    assert result1.id == crew_id
    assert result1.key is None

    crew2 = MockCrewOnlyKey(key="test-key")
    result2 = crew2.execute()
    assert result2 is not None
    assert result2.id is None
    assert result2.key == "test-key"


def test_with_crew_context_decorator_preserves_return_value():
    class MockCrew:
        def __init__(self, id, key):
            self.id = id
            self.key = key

        @with_crew_context
        def execute(self, value):
            context = get_crew_context()
            return {"value": value, "has_context": context is not None}

    crew = MockCrew(id=uuid.uuid4(), key="test")
    result = crew.execute("test-value")
    assert result["value"] == "test-value"
    assert result["has_context"] is True


def test_crew_context_exception_handling():
    crew_id = uuid.uuid4()

    with pytest.raises(ValueError):
        with crew_context(crew_id=crew_id, crew_key="test"):
            assert get_crew_context() is not None
            raise ValueError("Test exception")

    assert get_crew_context() is None

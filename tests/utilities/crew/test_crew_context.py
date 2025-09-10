import uuid

import pytest
from opentelemetry import baggage
from opentelemetry.context import attach, detach

from crewai.utilities.crew.crew_context import get_crew_context
from crewai.utilities.crew.models import CrewContext


def test_crew_context_creation():
    crew_id = str(uuid.uuid4())
    context = CrewContext(id=crew_id, key="test-crew")
    assert context.id == crew_id
    assert context.key == "test-crew"


def test_get_crew_context_with_baggage():
    crew_id = str(uuid.uuid4())
    assert get_crew_context() is None

    crew_ctx = CrewContext(id=crew_id, key="test-key")
    ctx = baggage.set_baggage("crew_context", crew_ctx)
    token = attach(ctx)

    try:
        context = get_crew_context()
        assert context is not None
        assert context.id == crew_id
        assert context.key == "test-key"
    finally:
        detach(token)

    assert get_crew_context() is None


def test_get_crew_context_empty():
    assert get_crew_context() is None


def test_baggage_nested_contexts():
    crew_id1 = str(uuid.uuid4())
    crew_id2 = str(uuid.uuid4())

    crew_ctx1 = CrewContext(id=crew_id1, key="outer")
    ctx1 = baggage.set_baggage("crew_context", crew_ctx1)
    token1 = attach(ctx1)

    try:
        outer_context = get_crew_context()
        assert outer_context.id == crew_id1
        assert outer_context.key == "outer"

        crew_ctx2 = CrewContext(id=crew_id2, key="inner")
        ctx2 = baggage.set_baggage("crew_context", crew_ctx2)
        token2 = attach(ctx2)

        try:
            inner_context = get_crew_context()
            assert inner_context.id == crew_id2
            assert inner_context.key == "inner"
        finally:
            detach(token2)

        restored_context = get_crew_context()
        assert restored_context.id == crew_id1
        assert restored_context.key == "outer"
    finally:
        detach(token1)

    assert get_crew_context() is None


def test_baggage_exception_handling():
    crew_id = str(uuid.uuid4())

    crew_ctx = CrewContext(id=crew_id, key="test")
    ctx = baggage.set_baggage("crew_context", crew_ctx)
    token = attach(ctx)

    with pytest.raises(ValueError):
        try:
            assert get_crew_context() is not None
            raise ValueError("Test exception")
        finally:
            detach(token)

    assert get_crew_context() is None

from collections.abc import Iterator
from contextlib import nullcontext
import logging

import pytest

from crewai.utilities.logger_utils import suppress_logging


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    """Provide a child logger and restore both loggers' original levels."""
    parent = logging.getLogger("crewai.tests.suppress_logging")
    child = logging.getLogger("crewai.tests.suppress_logging.child")
    parent_level, child_level = parent.level, child.level
    parent.setLevel(logging.INFO)
    child.setLevel(logging.INFO)
    try:
        yield child
    finally:
        child.setLevel(child_level)
        parent.setLevel(parent_level)


@pytest.mark.parametrize("error_type", [None, ValueError, KeyboardInterrupt])
def test_suppress_logging_restores_level(
    logger: logging.Logger, error_type: type[BaseException] | None
) -> None:
    """Restore the configured level after normal, error, and interrupted exits."""
    error = error_type("query interrupted") if error_type else None
    expected = pytest.raises(error_type) if error_type else nullcontext()

    with expected as raised:
        with suppress_logging(logger.name, logging.ERROR):
            assert not logger.isEnabledFor(logging.WARNING)
            if error is not None:
                raise error

    if error is not None:
        assert raised.value is error
    assert logger.isEnabledFor(logging.WARNING)
    assert logger.level == logging.INFO


@pytest.mark.parametrize("fail", [False, True])
def test_suppress_logging_preserves_inherited_level(
    logger: logging.Logger, fail: bool
) -> None:
    """Keep NOTSET inheritance so later parent-level changes remain effective."""
    logger.setLevel(logging.NOTSET)
    expected = pytest.raises(ValueError) if fail else nullcontext()

    with expected:
        with suppress_logging(logger.name, logging.ERROR):
            assert not logger.isEnabledFor(logging.WARNING)
            if fail:
                raise ValueError("query failed")

    assert logger.level == logging.NOTSET
    assert logger.parent is not None
    logger.parent.setLevel(logging.DEBUG)
    assert logger.isEnabledFor(logging.DEBUG)

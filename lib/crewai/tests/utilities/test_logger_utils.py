"""Regression tests for suppress_logging level restoration (issue #8214)."""

import logging

from crewai.utilities.logger_utils import suppress_logging


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def test_suppress_logging_restores_level_after_exception() -> None:
    logger = _make_logger("test-suppress-logging.exception")
    try:
        with suppress_logging(logger.name, logging.ERROR):
            raise ValueError("synthetic query failure")
    except ValueError:
        pass
    assert logger.level == logging.INFO


def test_suppress_logging_restores_level_on_clean_exit() -> None:
    logger = _make_logger("test-suppress-logging.clean")
    with suppress_logging(logger.name, logging.ERROR):
        assert logger.level == logging.ERROR
    assert logger.level == logging.INFO


def test_suppress_logging_propagates_exception() -> None:
    logger = _make_logger("test-suppress-logging.propagates")
    try:
        with suppress_logging(logger.name, logging.ERROR):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")


def test_suppress_logging_preserves_notset_inheritance() -> None:
    parent = logging.getLogger("test-suppress-logging.notset")
    child = logging.getLogger("test-suppress-logging.notset.child")
    parent.setLevel(logging.INFO)
    child.setLevel(logging.NOTSET)

    with suppress_logging(child.name, logging.ERROR):
        assert child.level == logging.ERROR

    assert child.level == logging.NOTSET
    parent.setLevel(logging.DEBUG)
    assert child.isEnabledFor(logging.DEBUG)

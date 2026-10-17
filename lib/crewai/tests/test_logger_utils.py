import logging
import pytest
from crewai.utilities.logger_utils import suppress_logging


def test_suppress_logging_restores_level_on_exception():
    parent = logging.getLogger("crewai_test_repro_parent")
    logger = logging.getLogger("crewai_test_repro_parent.child")
    parent.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    try:
        with suppress_logging(logger.name, logging.ERROR):
            assert logger.level == logging.ERROR
            raise ValueError("simulated query error")
    except ValueError:
        pass

    assert logger.level == logging.INFO


def test_suppress_logging_preserves_notset_inheritance():
    parent = logging.getLogger("crewai_test_notset_parent")
    logger = logging.getLogger("crewai_test_notset_parent.child")
    parent.setLevel(logging.INFO)
    logger.setLevel(logging.NOTSET)

    with suppress_logging(logger.name, logging.ERROR):
        assert logger.level == logging.ERROR

    assert logger.level == logging.NOTSET
    parent.setLevel(logging.DEBUG)
    assert logger.isEnabledFor(logging.DEBUG) is True

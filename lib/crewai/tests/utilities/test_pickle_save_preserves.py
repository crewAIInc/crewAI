"""A failed save must not truncate a previously saved file."""

import pickle
from pathlib import Path
from threading import Lock

import pytest

from crewai.utilities.training_handler import CrewTrainingHandler


def test_failed_save_preserves_previous_data(tmp_path: Path):
    handler = CrewTrainingHandler(str(tmp_path / "training.pkl"))
    handler.save_trained_data("saved-agent", {"score": 10})

    with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
        handler.save_trained_data("new-agent", {"lock": Lock()})

    assert handler.load()["saved-agent"]["score"] == 10

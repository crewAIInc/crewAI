"""Tests for the non-UUID flow state id warning (CrewAI AMP parity).

CrewAI AMP deployments reject non-UUID ``inputs.id`` values with HTTP 422,
while the OSS library accepts any string locally. A persisted flow kicked off
with a non-UUID ``id`` should surface a one-time warning so the mismatch is
caught before deployment — without changing local behavior.
"""

import os
import warnings
from uuid import uuid4

from crewai.flow.flow import Flow, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


AMP_WARNING_MATCH = "CrewAI AMP"


def _build_persisted_flow_class(persistence):
    class PersistedFlow(Flow[dict]):
        initial_state = dict

        @start()
        @persist(persistence)
        def init_step(self):
            self.state["message"] = "ran"

    return PersistedFlow


def test_non_uuid_id_with_persistence_warns_once_and_still_runs(tmp_path):
    """A non-UUID `id` input on a persisted flow warns once but runs fine."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    flow = _build_persisted_flow_class(persistence)(persistence=persistence)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow.kickoff(inputs={"id": "session-1"})

    amp_warnings = [
        w for w in caught if AMP_WARNING_MATCH in str(w.message)
    ]
    assert len(amp_warnings) == 1, (
        f"expected exactly one AMP parity warning, got {len(amp_warnings)}"
    )
    assert issubclass(amp_warnings[0].category, UserWarning)
    message = str(amp_warnings[0].message)
    assert "422" in message
    assert "UUID" in message

    # Behavior is unchanged: the flow ran and kept the non-UUID id locally.
    assert flow.state["id"] == "session-1"
    assert flow.state["message"] == "ran"
    assert persistence.load_state("session-1") is not None


def test_valid_uuid_id_with_persistence_does_not_warn(tmp_path):
    """A valid UUID `id` input on a persisted flow emits no AMP warning."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    flow = _build_persisted_flow_class(persistence)(persistence=persistence)
    valid_id = str(uuid4())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow.kickoff(inputs={"id": valid_id})

    amp_warnings = [
        w for w in caught if AMP_WARNING_MATCH in str(w.message)
    ]
    assert amp_warnings == []
    assert flow.state["id"] == valid_id
    assert flow.state["message"] == "ran"


def test_non_uuid_id_with_method_level_persist_only_warns(tmp_path):
    """Method-level @persist (no instance/class persistence) still warns.

    Method-level @persist never sets `flow.persistence`, but the flow still
    saves state under the provided id, so the AMP parity warning must fire.
    """
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    # No persistence= constructor arg: only the method is persisted.
    flow = _build_persisted_flow_class(persistence)()
    assert flow.persistence is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow.kickoff(inputs={"id": "session-1"})

    amp_warnings = [
        w for w in caught if AMP_WARNING_MATCH in str(w.message)
    ]
    assert len(amp_warnings) == 1, (
        f"expected exactly one AMP parity warning, got {len(amp_warnings)}"
    )

    # Behavior is unchanged: the method-level @persist saved the state.
    assert flow.state["id"] == "session-1"
    assert flow.state["message"] == "ran"
    assert persistence.load_state("session-1") is not None


def test_non_uuid_id_without_persistence_does_not_warn():
    """Without persistence, a non-UUID `id` input emits no AMP warning."""

    class PlainFlow(Flow[dict]):
        initial_state = dict

        @start()
        def init_step(self):
            self.state["message"] = "ran"

    flow = PlainFlow()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flow.kickoff(inputs={"id": "session-1"})

    amp_warnings = [
        w for w in caught if AMP_WARNING_MATCH in str(w.message)
    ]
    assert amp_warnings == []
    assert flow.state["id"] == "session-1"
    assert flow.state["message"] == "ran"

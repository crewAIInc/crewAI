"""Tests for SQLiteFlowPersistence JSON serialization fix (issue #7358)."""

import json
import tempfile
from datetime import datetime, timezone
import uuid

import pytest
from pydantic import BaseModel, Field

from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


class StateWithSpecialTypes(BaseModel):
    """State model with types that need special JSON serialization."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    tags: set[str] = Field(default_factory=lambda: {"analytics", "v1"})


class TestToStateDictJsonMode:
    """_to_state_dict must produce JSON-serializable dicts."""

    def test_pydantic_model_dump_json_mode(self):
        state = StateWithSpecialTypes()
        result = SQLiteFlowPersistence._to_state_dict(state)
        # Should be JSON-serializable without errors
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        # datetime should be a string in JSON mode
        assert isinstance(result["created_at"], str)

    def test_dict_passthrough(self):
        d = {"key": "value", "count": 42}
        result = SQLiteFlowPersistence._to_state_dict(d)
        assert result == d

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="state_data must be"):
            SQLiteFlowPersistence._to_state_dict("not a dict")


class TestSaveStateSql:
    """_save_state_sql must handle non-JSON-native types gracefully."""

    def test_save_with_special_types(self):
        """Saving state with datetime/UUID/set should not crash."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            persistence = SQLiteFlowPersistence(f.name)
            state = StateWithSpecialTypes()
            state_dict = SQLiteFlowPersistence._to_state_dict(state)

            import sqlite3

            with sqlite3.connect(f.name) as conn:
                persistence._save_state_sql(
                    conn, "test-flow", "step_one", state_dict
                )
                # Verify we can read it back
                cursor = conn.execute(
                    "SELECT state_json FROM flow_states WHERE flow_uuid = ?",
                    ("test-flow",),
                )
                row = cursor.fetchone()
                assert row is not None
                loaded = json.loads(row[0])
                assert "created_at" in loaded
                assert "user_id" in loaded

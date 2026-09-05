"""The tools must keep Velaris's guarantee: a budget is enforced."""

import json

import pytest


pytest.importorskip("velaris")

from crewai_tools.tools.velaris_tool import VelarisAuditTool, VelarisRunTool


READS_A_FILE = """
fn peek(path: Text) -> Text uses fs or fail {
    return try read_file(path)
}

fn main() uses io, fs {
    print("start")
    check peek("__PATH__") {
        ok body {
            print("READ IT")
        }
        fail why {
            print("failed")
        }
    }
}
"""

PURE = """
fn main() uses io {
    print(6 * 7)
}
"""


@pytest.fixture
def reads_a_file(tmp_path):
    """READS_A_FILE pointed at a file that exists."""
    note = tmp_path / "note.txt"
    note.write_text("hello")
    return READS_A_FILE.replace("__PATH__", str(note).replace("\\", "/"))


def test_audit_names_every_effect():
    report = json.loads(VelarisAuditTool().run(source=READS_A_FILE))
    assert report["schema"] == "velaris.audit/1"
    assert report["effects"] == ["fs", "io"]


def test_run_refuses_an_effect_outside_the_budget():
    out = VelarisRunTool(allow=["io"]).run(source=READS_A_FILE)
    assert "REFUSED" in out and "'fs'" in out
    assert "READ IT" not in out


def test_run_permits_what_the_budget_allows(reads_a_file):
    out = VelarisRunTool(allow=["io", "fs"]).run(source=reads_a_file)
    assert "READ IT" in out


def test_run_returns_output():
    assert VelarisRunTool(allow=["io"]).run(source=PURE).strip() == "42"


def test_the_default_budget_is_io_only():
    assert VelarisRunTool().allow == ["io"]

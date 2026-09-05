"""The tools must keep Velaris's guarantee: a budget is enforced."""

import json
import sys

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

FOREVER = """
fn main() uses io {
    let i = 0
    while i >= 0 {
        i = i + 1
        if i > 1000000 {
            i = 0
        }
    }
    print("REACHED THE END")
}
"""

EATS_MEMORY = """
fn main() uses io {
    let s = "xxxxxxxxxxxxxxxx"
    let i = 0
    while i < 40 {
        s = s + s
        i = i + 1
    }
    print(length(s))
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
    # a refusal is not a failure the program can catch: the fail branch
    # must not run either
    assert "failed" not in out


def test_run_stops_a_program_that_never_ends():
    out = VelarisRunTool(allow=["io"], timeout=2).run(source=FOREVER)
    assert "STOPPED" in out
    # the marker only the program could print must be absent; the
    # compiler's own hint text mentions a loop that "never ends"
    assert "REACHED THE END" not in out


@pytest.mark.skipif(
    sys.platform == "win32", reason="the memory cap is enforced on Linux and macOS"
)
def test_run_stops_a_program_that_eats_memory():
    out = VelarisRunTool(allow=["io"], max_memory_mb=150, timeout=60).run(
        source=EATS_MEMORY
    )
    assert "used more than 150" in out
    # the program must not have printed a length (16 * 2**40 starts with 16)
    assert not any(line.startswith("16") for line in out.splitlines())


def test_the_default_limits_are_set():
    tool = VelarisRunTool()
    assert tool.timeout == 30.0 and tool.max_memory_mb == 512


def test_run_permits_what_the_budget_allows(reads_a_file):
    out = VelarisRunTool(allow=["io", "fs"]).run(source=reads_a_file)
    assert "READ IT" in out


def test_run_returns_output():
    assert VelarisRunTool(allow=["io"]).run(source=PURE).strip() == "42"


def test_the_default_budget_is_io_only():
    assert VelarisRunTool().allow == ["io"]

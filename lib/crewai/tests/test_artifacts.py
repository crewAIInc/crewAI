import json
from pathlib import Path
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from crewai import Agent, Artifact, Crew, Task, save_artifact
from crewai.artifacts import (
    LocalArtifactStore,
    SaveArtifactTool,
    get_artifact_store,
    set_artifact_store,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.artifact_events import ArtifactCreatedEvent
from crewai.execution import (
    clear_execution_uuid,
    get_execution_uuid,
    set_execution_uuid,
)
from crewai.flow import Flow, start


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    local = LocalArtifactStore(tmp_path / "artifacts")
    set_artifact_store(local)
    token = set_execution_uuid("exec-1")
    yield local
    clear_execution_uuid(token)
    set_artifact_store(original)


def _agent(**kwargs: Any) -> Agent:
    return Agent(role="Writer", goal="Write", backstory="Writes reports", **kwargs)


def test_same_identifier_creates_new_versions_and_emits_events(store, tmp_path):
    events: list[ArtifactCreatedEvent] = []
    report = Artifact(identifier="q3-report", title="Q3 report", kind="html")

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(ArtifactCreatedEvent)(lambda _, e: events.append(e))
        first = save_artifact(report, content="<h1>v1</h1>")
        second = save_artifact(report, content="<h1>v2</h1>")
        crewai_event_bus.flush()

    directory = tmp_path / "artifacts" / "exec-1" / "q3-report"
    assert (first.version, second.version) == (1, 2)
    assert (directory / "v1.html").read_text() == "<h1>v1</h1>"
    assert (directory / "v2.html").read_text() == "<h1>v2</h1>"
    assert second.content_type == "text/html"
    assert second.url == (directory / "v2.html").resolve().as_uri()
    assert [(e.id, e.identifier, e.version) for e in events] == [
        (first.id, "q3-report", 1),
        (second.id, "q3-report", 2),
    ]
    assert first.id != second.id
    assert uuid.UUID(first.id).version == 4


def test_tool_saves_file_and_returns_link(store, tmp_path):
    (tmp_path / "chart.png").write_bytes(b"\x89PNG")

    result = json.loads(
        SaveArtifactTool().run(
            identifier="chart", title="Chart", kind="image", file_path="chart.png"
        )
    )

    assert result["version"] == 1
    assert uuid.UUID(result["id"])
    assert result["share_url"] is None
    saved = tmp_path / "artifacts" / "exec-1" / "chart" / "v1.png"
    assert result["url"] == saved.resolve().as_uri()
    assert saved.read_bytes() == b"\x89PNG"


def test_tool_saves_a_rewritten_file_again_when_tool_caching_is_on(store, tmp_path):
    from types import SimpleNamespace

    from crewai.agents.cache.cache_handler import CacheHandler
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.utilities.agent_utils import execute_single_native_tool_call

    tool = SaveArtifactTool()
    handler = ToolsHandler(cache=CacheHandler())
    agent = _agent()
    chart = tmp_path / "chart.svg"
    tool_call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(
            name="save_artifact",
            arguments=json.dumps(
                {"identifier": "chart", "title": "Chart", "kind": "svg", "file_path": "chart.svg"}
            ),
        ),
    )

    results = []
    for label in ("draft", "final"):
        chart.write_text(f"<svg>{label}</svg>")
        results.append(
            execute_single_native_tool_call(
                tool_call,
                available_functions={"save_artifact": tool.run},
                original_tools=[tool],
                structured_tools=[tool.to_structured_tool()],
                tools_handler=handler,
                agent=agent,
                task=None,
                crew=None,
                event_source=agent,
            )
        )

    directory = tmp_path / "artifacts" / "exec-1" / "chart"
    assert [json.loads(r.result)["version"] for r in results] == [1, 2]
    assert (directory / "v2.svg").read_text() == "<svg>final</svg>"


def test_tool_refuses_files_outside_workdir(store):
    result = SaveArtifactTool().run(
        identifier="secret", title="x", kind="file", file_path="/etc/hosts"
    )

    assert result.startswith("Error saving artifact")
    assert not (store.root / "exec-1" / "secret").exists()


def test_rejects_bad_identifier():
    with pytest.raises(ValueError):
        Artifact(identifier="Bad Id", title="x", kind="text")


def test_task_artifact_saves_final_output(store, tmp_path):
    agent = _agent()
    task = Task(
        description="Write the report",
        expected_output="HTML",
        agent=agent,
        artifact=Artifact(identifier="report", title="Report", kind="html"),
    )

    with patch.object(Agent, "execute_task", return_value="<p>done</p>"):
        task.execute_sync(agent=agent)

    saved = tmp_path / "artifacts" / "exec-1" / "report" / "v1.html"
    assert saved.read_text() == "<p>done</p>"


def test_task_artifact_failure_does_not_fail_task(store):
    agent = _agent()
    task = Task(
        description="Write the report",
        expected_output="HTML",
        agent=agent,
        artifact=Artifact(identifier="report", title="Report", kind="html"),
    )

    with (
        patch.object(Agent, "execute_task", return_value="<p>done</p>"),
        patch.object(store, "save", side_effect=OSError("disk full")),
    ):
        output = task.execute_sync(agent=agent)

    assert output.raw == "<p>done</p>"
    assert output.artifact is None


def test_task_output_exposes_saved_artifact(store, tmp_path):
    agent = _agent()
    seen = []
    task = Task(
        description="Write the report",
        expected_output="HTML",
        agent=agent,
        artifact=Artifact(identifier="report", title="Report", kind="html"),
        # A callback is where people post the link onward.
        callback=lambda out: seen.append(out.artifact.url),
    )

    with patch.object(Agent, "execute_task", return_value="<p>done</p>"):
        output = task.execute_sync(agent=agent)

    saved = tmp_path / "artifacts" / "exec-1" / "report" / "v1.html"
    assert output.artifact.version == 1
    assert output.artifact.url == saved.resolve().as_uri()
    assert seen == [output.artifact.url]


@pytest.mark.parametrize("kind", ["image", "pdf", "file"])
def test_task_rejects_kinds_text_output_cannot_fill(kind):
    with pytest.raises(ValueError, match=f"{kind}.*file_path"):
        Task(
            description="d",
            expected_output="o",
            artifact=Artifact(identifier="report", title="Report", kind=kind),
        )


@pytest.mark.parametrize(
    ("crew_flag", "agent_flag", "expected"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_artifacts_flag_adds_tool(crew_flag, agent_flag, expected):
    agent = _agent(artifacts=agent_flag)
    task = Task(description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], artifacts=crew_flag)

    names = [t.name for t in crew._prepare_tools(agent, task, [])]

    assert ("save_artifact" in names) is expected


def test_flow_step_saves_under_the_flow_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    set_artifact_store(LocalArtifactStore(tmp_path / "artifacts"))

    class PostFlow(Flow):
        @start()
        def save_content(self):
            saved = save_artifact(
                Artifact(identifier="post", title="Post", kind="markdown"),
                content="# Hello",
            )
            return saved, get_execution_uuid()

    try:
        saved, run_id = PostFlow(tracing=False).kickoff()
    finally:
        set_artifact_store(original)

    path = tmp_path / "artifacts" / run_id / "post" / "v1.md"
    assert path.read_text() == "# Hello"
    assert saved.url == path.resolve().as_uri()


def _capture_events(run):
    events: list[ArtifactCreatedEvent] = []
    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(ArtifactCreatedEvent)(lambda _, e: events.append(e))
        run()
        crewai_event_bus.flush()
    return events


def test_task_artifact_event_names_task_and_agent(store):
    agent = _agent()
    task = Task(
        name="Report",
        description="Write the report",
        expected_output="HTML",
        agent=agent,
        artifact=Artifact(identifier="report", title="Report", kind="html"),
    )

    with patch.object(Agent, "execute_task", return_value="<p>done</p>"):
        events = _capture_events(lambda: task.execute_sync(agent=agent))

    assert [(e.task_id, e.task_name, e.agent_role) for e in events] == [
        (str(task.id), "Report", "Writer")
    ]


def test_tool_event_names_task_and_agent(store):
    agent = _agent()
    task = Task(name="Chart", description="d", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], artifacts=True)
    tool = next(
        t for t in crew._prepare_tools(agent, task, []) if t.name == "save_artifact"
    )

    events = _capture_events(
        lambda: tool.run(identifier="chart", title="Chart", kind="text", content="x")
    )

    assert [(e.task_id, e.agent_role) for e in events] == [(str(task.id), "Writer")]


def _kickoff_patches(raw: str = "<p>done</p>"):
    from crewai.lite_agent_output import LiteAgentOutput

    out = LiteAgentOutput(raw=raw, agent_role="Writer")

    async def _async(*_a, **_k):
        return out

    return (
        patch.object(Agent, "_execute_and_build_output", return_value=out),
        patch.object(Agent, "_execute_and_build_output_async", side_effect=_async),
    )


def test_kickoff_saves_output_as_artifact(store, tmp_path):
    sync_patch, _ = _kickoff_patches()
    report = Artifact(identifier="report", title="Report", kind="html")

    with sync_patch:
        events = []
        result = None

        def run():
            nonlocal result
            result = _agent().kickoff("Write it", artifact=report)

        events = _capture_events(run)

    saved = tmp_path / "artifacts" / "exec-1" / "report" / "v1.html"
    assert saved.read_text() == "<p>done</p>"
    assert result.artifact.url == saved.resolve().as_uri()
    assert [e.agent_role for e in events] == ["Writer"]


def test_kickoff_async_saves_output_as_artifact(store, tmp_path):
    import asyncio

    _, async_patch = _kickoff_patches()
    report = Artifact(identifier="report", title="Report", kind="html")

    with async_patch:
        result = asyncio.run(_agent().kickoff_async("Write it", artifact=report))

    assert result.artifact.version == 1
    assert (tmp_path / "artifacts" / "exec-1" / "report" / "v1.html").exists()


def test_kickoff_rejects_file_kinds_before_running(store):
    sync_patch, _ = _kickoff_patches()

    with sync_patch as execute, pytest.raises(ValueError, match="image"):
        _agent().kickoff(
            "Draw it", artifact=Artifact(identifier="c", title="C", kind="image")
        )

    execute.assert_not_called()


def test_kickoff_artifact_failure_does_not_fail_kickoff(store):
    sync_patch, _ = _kickoff_patches()

    with sync_patch, patch.object(store, "save", side_effect=OSError("disk full")):
        result = _agent().kickoff(
            "Write it", artifact=Artifact(identifier="r", title="R", kind="html")
        )

    assert result.raw == "<p>done</p>"
    assert result.artifact is None


@pytest.mark.parametrize(
    ("content", "kind"),
    [
        ("<!DOCTYPE html><html></html>", "html"),
        ("  <html><body>x</body></html>", "html"),
        ('<svg xmlns="http://www.w3.org/2000/svg"></svg>', "svg"),
        ('{"a": 1}', "json"),
        ("[1, 2]", "json"),
        ("# Title\n\nText", "markdown"),
        ("plain words", "markdown"),
    ],
)
def test_guess_kind(content, kind):
    from crewai.artifacts import guess_kind

    assert guess_kind(content) == kind


def test_task_artifact_true_names_it_after_the_task(store, tmp_path):
    agent = _agent()
    task = Task(
        name="Q3 report",
        description="Write the report",
        expected_output="HTML",
        agent=agent,
        artifact=True,
    )

    with patch.object(Agent, "execute_task", return_value="<!doctype html><p>x</p>"):
        output = task.execute_sync(agent=agent)

    saved = output.artifact
    assert (saved.identifier, saved.title, saved.kind) == ("q3-report", "Q3 report", "html")
    assert (tmp_path / "artifacts" / "exec-1" / "q3-report" / "v1.html").exists()


def test_kickoff_artifact_true_names_it_after_the_agent(store):
    sync_patch, _ = _kickoff_patches("# Notes")

    with sync_patch:
        result = Agent(
            role="Research Analyst", goal="g", backstory="b"
        ).kickoff("Write it", artifact=True)

    saved = result.artifact
    assert (saved.identifier, saved.title, saved.kind) == (
        "research-analyst",
        "Research Analyst",
        "markdown",
    )


def test_kickoff_artifact_true_in_a_flow_is_named_after_the_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    set_artifact_store(LocalArtifactStore(tmp_path / "artifacts"))
    sync_patch, _ = _kickoff_patches("<!doctype html><p>x</p>")

    class ReportFlow(Flow):
        @start()
        def write_report(self):
            return _agent().kickoff("Write it", artifact=True)

    try:
        with sync_patch:
            result = ReportFlow(tracing=False).kickoff()
    finally:
        set_artifact_store(original)

    saved = result.artifact
    assert (saved.identifier, saved.title, saved.kind) == (
        "write-report",
        "Write report",
        "html",
    )


def test_declarative_agent_action_saves_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    set_artifact_store(LocalArtifactStore(tmp_path / "artifacts"))
    _, async_patch = _kickoff_patches("<!doctype html><p>x</p>")
    yaml_str = """
schema: crewai.flow/v1
name: ReportFlow
methods:
  write_report:
    start: true
    do:
      call: agent
      with:
        role: Analyst
        goal: Write the report
        backstory: Writes reports.
        input: "Write it"
        artifact: true
"""

    try:
        with async_patch:
            result = Flow.from_declaration(contents=yaml_str).kickoff()
    finally:
        set_artifact_store(original)

    assert (result.artifact.identifier, result.artifact.kind) == ("write-report", "html")


def _writes(path, data: bytes = b"%PDF-1.7 fake", raw: str = "Done."):
    """Stand-in for an agent whose tool writes ``path`` during the run."""

    def run(*_a, **_k):
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
        return raw

    return run


def test_task_artifact_saves_file_the_agent_wrote(store, tmp_path):
    agent = _agent()
    task = Task(
        description="Build the {quarter} report as a PDF at output/{quarter}.pdf",
        expected_output="Done",
        agent=agent,
        artifact=Artifact(
            identifier="report", title="Report", kind="pdf", file_path="output/{quarter}.pdf"
        ),
    )
    task.interpolate_inputs_and_add_conversation_history({"quarter": "q3"})

    with patch.object(Agent, "execute_task", side_effect=_writes("output/q3.pdf")):
        output = task.execute_sync(agent=agent)

    saved = tmp_path / "artifacts" / "exec-1" / "report" / "v1.pdf"
    assert saved.read_bytes() == b"%PDF-1.7 fake"
    assert output.artifact.content_type == "application/pdf"
    assert output.raw == "Done."


def test_task_artifact_skips_missing_file(store):
    agent = _agent()
    task = Task(
        description="d",
        expected_output="o",
        agent=agent,
        artifact=Artifact(identifier="r", title="R", kind="pdf", file_path="nope.pdf"),
    )

    with patch.object(Agent, "execute_task", return_value="Done."):
        output = task.execute_sync(agent=agent)

    assert output.artifact is None


def test_task_artifact_accepts_file_from_an_earlier_run(store, tmp_path):
    import os

    old = tmp_path / "chart.png"
    old.write_bytes(b"\x89PNG old")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    agent = _agent()
    task = Task(
        description="d",
        expected_output="o",
        agent=agent,
        artifact=Artifact(identifier="c", title="C", kind="image", file_path="chart.png"),
    )

    with patch.object(Agent, "execute_task", return_value="Done."):
        output = task.execute_sync(agent=agent)

    assert output.artifact is not None
    assert (tmp_path / "artifacts" / "exec-1" / "c" / "v1.png").read_bytes() == b"\x89PNG old"


def test_file_kind_without_file_path_is_rejected():
    with pytest.raises(ValueError, match="file_path"):
        Task(
            description="d",
            expected_output="o",
            artifact=Artifact(identifier="c", title="C", kind="image"),
        )


def test_kickoff_artifact_saves_file_the_agent_wrote(store, tmp_path):
    from crewai.lite_agent_output import LiteAgentOutput

    def run(*_a, **_k):
        _writes("chart.png", b"\x89PNG new")()
        return LiteAgentOutput(raw="Drawn.", agent_role="Writer")

    with patch.object(Agent, "_execute_and_build_output", side_effect=run):
        result = _agent().kickoff(
            "Draw it",
            artifact=Artifact(identifier="chart", title="Chart", kind="image", file_path="chart.png"),
        )

    assert result.artifact.content_type == "image/png"
    assert (tmp_path / "artifacts" / "exec-1" / "chart" / "v1.png").read_bytes() == b"\x89PNG new"


def test_declarative_agent_artifact_file_path_uses_flow_expressions(tmp_path, monkeypatch):
    from crewai.lite_agent_output import LiteAgentOutput

    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    set_artifact_store(LocalArtifactStore(tmp_path / "artifacts"))

    async def run(*_a, **_k):
        _writes("output/q3.png", b"\x89PNG q3")()
        return LiteAgentOutput(raw="Drawn.", agent_role="Designer")

    yaml_str = """
schema: crewai.flow/v1
name: ChartFlow
state:
  type: dict
  default:
    quarter: q3
methods:
  draw_chart:
    start: true
    do:
      call: agent
      with:
        role: Designer
        goal: Draw charts
        backstory: Makes charts.
        input: "Draw the ${state.quarter} chart"
        artifact:
          identifier: sales-chart
          title: Sales chart
          kind: image
          file_path: "output/${state.quarter}.png"
"""

    try:
        with patch.object(Agent, "_execute_and_build_output_async", side_effect=run):
            result = Flow.from_declaration(contents=yaml_str).kickoff()
    finally:
        set_artifact_store(original)

    assert result.artifact.identifier == "sales-chart"
    assert result.artifact.content_type == "image/png"


def test_artifact_can_point_at_the_task_output_file(store, tmp_path):
    agent = _agent()
    task = Task(
        description="d",
        expected_output="HTML",
        agent=agent,
        output_file="report.html",
        artifact=Artifact(identifier="r", title="R", kind="html", file_path="report.html"),
    )

    with patch.object(Agent, "execute_task", return_value="<html>x</html>"):
        output = task.execute_sync(agent=agent)

    assert output.artifact is not None
    assert (tmp_path / "artifacts" / "exec-1" / "r" / "v1.html").read_text() == "<html>x</html>"


@pytest.mark.parametrize("kind", ["image", "pdf", "file"])
def test_tool_refuses_text_for_file_kinds(store, kind):
    result = SaveArtifactTool().run(identifier="p", title="P", kind=kind, content="# text")

    assert result.startswith("Error saving artifact")
    assert not (store.root / "exec-1" / "p").exists()


def test_pdf_bytes_from_code_are_saved(store):
    saved = save_artifact(Artifact(identifier="p", title="P", kind="pdf"), content=b"%PDF-1.7")

    assert saved.content_type == "application/pdf"


def test_standalone_task_gives_artifacts_agent_the_tool(store):
    agent = _agent(artifacts=True)
    seen = {}

    def spy(self, task, context=None, tools=None):
        self.create_agent_executor(tools=tools, task=task)
        seen["tools"] = [t.name for t in self.agent_executor.tools]
        return "ok"

    with patch.object(Agent, "execute_task", spy):
        Task(description="d", expected_output="o", agent=agent).execute_sync()

    assert "save_artifact" in seen["tools"]


def test_declarative_file_path_from_state_cannot_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = get_artifact_store()
    set_artifact_store(LocalArtifactStore(tmp_path / "artifacts"))
    _, async_patch = _kickoff_patches()
    yaml_str = """
schema: crewai.flow/v1
name: ChartFlow
state:
  type: dict
  default:
    name: "../../etc/passwd"
methods:
  draw:
    start: true
    do:
      call: agent
      with:
        role: Designer
        goal: g
        backstory: b
        input: Draw it
        artifact:
          identifier: chart
          title: Chart
          kind: image
          file_path: "out/${state.name}.png"
"""

    try:
        with async_patch, pytest.raises(Exception, match="file_path"):
            Flow.from_declaration(contents=yaml_str).kickoff()
    finally:
        set_artifact_store(original)


def test_async_kickoff_saves_off_the_event_loop(store):
    import asyncio
    import threading

    _, async_patch = _kickoff_patches()
    loop_thread = {}
    save_thread = {}
    real_save = store.save

    def spy(*a, **k):
        save_thread["id"] = threading.get_ident()
        return real_save(*a, **k)

    async def run():
        loop_thread["id"] = threading.get_ident()
        return await _agent().kickoff_async(
            "Write it", artifact=Artifact(identifier="r", title="R", kind="html")
        )

    with async_patch, patch.object(store, "save", side_effect=spy):
        result = asyncio.run(run())

    assert result.artifact is not None
    assert save_thread["id"] != loop_thread["id"]


def test_async_task_saves_off_the_event_loop(store):
    import asyncio
    import threading

    agent = _agent()
    task = Task(
        description="d",
        expected_output="o",
        agent=agent,
        artifact=Artifact(identifier="r", title="R", kind="html"),
    )
    threads = {}
    real_save = store.save

    def spy(*a, **k):
        threads["save"] = threading.get_ident()
        return real_save(*a, **k)

    async def run():
        threads["loop"] = threading.get_ident()
        return await task.aexecute_sync(agent=agent)

    async def fake_aexecute(*_a, **_k):
        return "<p>x</p>"

    with (
        patch.object(Agent, "aexecute_task", side_effect=fake_aexecute),
        patch.object(store, "save", side_effect=spy),
    ):
        output = asyncio.run(run())

    assert output.artifact is not None
    assert threads["save"] != threads["loop"]


def test_local_store_never_reuses_a_version(store):
    """Another process may claim a version between our count and our write."""
    report = Artifact(identifier="r", title="R", kind="html")
    directory = store.root / "exec-1" / "r"
    directory.mkdir(parents=True)
    real_glob = type(directory).glob

    def stale_glob(self, pattern):
        # Pretend the other process's v1 is not visible yet when we count.
        return [] if self == directory else real_glob(self, pattern)

    save_artifact(report, content="mine")
    with patch.object(type(directory), "glob", stale_glob):
        second = save_artifact(report, content="theirs")

    assert second.version == 2
    assert (directory / "v1.html").read_text() == "mine"


def _a2a_agent() -> Agent:
    from crewai.a2a.config import A2AClientConfig

    return Agent(
        role="Writer",
        goal="g",
        backstory="b",
        a2a=A2AClientConfig(endpoint="http://example.com"),
    )


def test_a2a_agent_kickoff_async_accepts_artifact(store):
    import asyncio

    agent = _a2a_agent()
    no_delegation = patch(
        "crewai.a2a.wrapper.get_a2a_agents_and_response_model", return_value=([], None)
    )

    with _kickoff_patches()[1], no_delegation:
        result = asyncio.run(
            agent.kickoff_async("hi", artifact=Artifact(identifier="r", title="R", kind="html"))
        )
    # A fresh stub: the first one's output object now carries an artifact.
    with _kickoff_patches()[1], no_delegation:
        plain = asyncio.run(agent.akickoff("hi"))

    assert result.artifact.version == 1
    assert plain.artifact is None


def test_a2a_delegation_saves_artifact_once_on_final_output(store, tmp_path):
    import asyncio

    from crewai.lite_agent_output import LiteAgentOutput

    agent = _a2a_agent()

    async def delegated(**_k):
        return LiteAgentOutput(raw="<p>final</p>", agent_role="Writer")

    with (
        patch(
            "crewai.a2a.wrapper.get_a2a_agents_and_response_model",
            return_value=([object()], None),
        ),
        patch("crewai.a2a.wrapper._akickoff_with_a2a", side_effect=delegated),
    ):
        result = asyncio.run(
            agent.kickoff_async("hi", artifact=Artifact(identifier="r", title="R", kind="html"))
        )

    assert result.artifact.version == 1
    assert sorted(p.name for p in (tmp_path / "artifacts" / "exec-1" / "r").iterdir()) == ["v1.html"]


def test_kickoff_artifact_failure_is_logged_without_verbose(store, caplog):
    sync_patch, _ = _kickoff_patches()

    with sync_patch, patch.object(store, "save", side_effect=OSError("disk full")):
        _agent().kickoff("Write it", artifact=Artifact(identifier="r", title="R", kind="html"))

    assert "disk full" in caplog.text


def test_task_artifact_file_path_is_checked_when_defined():
    with pytest.raises(ValueError, match="artifact.file_path") as error:
        Task(
            description="d",
            expected_output="o",
            artifact=Artifact(identifier="c", title="C", kind="image", file_path="../c.png"),
        )
    assert "output_file" not in str(error.value)


@pytest.mark.parametrize(
    ("kind", "data", "content_type", "suffix"),
    [
        ("image", b"\x89PNG\r\n\x1a\nrest", "image/png", ".png"),
        ("image", b"\xff\xd8\xff\xe0rest", "image/jpeg", ".jpg"),
        ("image", b"GIF89arest", "image/gif", ".gif"),
        ("image", b"RIFF\x00\x00\x00\x00WEBPrest", "image/webp", ".webp"),
        ("file", b"\x00\x01\x02", "application/octet-stream", ".bin"),
    ],
)
def test_bytes_from_code_for_every_file_kind(store, kind, data, content_type, suffix):
    saved = save_artifact(Artifact(identifier="b", title="B", kind=kind), content=data)

    assert saved.content_type == content_type
    assert saved.url.endswith("/v1" + suffix)


def test_unknown_image_bytes_are_refused(store):
    with pytest.raises(ValueError, match="image"):
        save_artifact(Artifact(identifier="b", title="B", kind="image"), content=b"not an image")


@pytest.mark.parametrize("data", [b"<svg xmlns='http://www.w3.org/2000/svg'/>", b"not an image"])
def test_unknown_image_file_bytes_are_refused(store, tmp_path, data):
    path = tmp_path / "chart.svg"
    path.write_bytes(data)

    with pytest.raises(ValueError, match="image bytes must be PNG, JPEG, GIF or WebP"):
        save_artifact(
            Artifact(identifier="b", title="B", kind="image"), file_path=str(path)
        )


def test_image_file_format_comes_from_its_bytes(store, tmp_path):
    path = tmp_path / "chart.txt"
    path.write_bytes(b"\x89PNG\r\n\x1a\nrest")

    saved = save_artifact(
        Artifact(identifier="b", title="B", kind="image"), file_path=str(path)
    )

    assert saved.content_type == "image/png"
    assert saved.url.endswith("/v1.png")


FENCED_HTML = '```html\n<!doctype html>\n<html lang="en">\n<body>Q3</body>\n</html>\n```'


def test_artifact_true_unwraps_a_fenced_reply(store, tmp_path):
    sync_patch, _ = _kickoff_patches(FENCED_HTML)

    with sync_patch:
        result = _agent().kickoff("Write it", artifact=True)

    saved = result.artifact
    assert saved.kind == "html"
    path = tmp_path / "artifacts" / "exec-1" / "writer" / "v1.html"
    assert path.read_text().startswith("<!doctype html>")
    assert "```" not in path.read_text()
    # The agent's own output is left as the model wrote it.
    assert result.raw == FENCED_HTML


def test_explicit_kind_unwraps_a_fenced_reply(store, tmp_path):
    sync_patch, _ = _kickoff_patches(FENCED_HTML)

    with sync_patch:
        _agent().kickoff(
            "Write it", artifact=Artifact(identifier="r", title="R", kind="html")
        )

    assert (tmp_path / "artifacts" / "exec-1" / "r" / "v1.html").read_text().startswith(
        "<!doctype html>"
    )


def test_fence_language_picks_the_kind(store):
    sync_patch, _ = _kickoff_patches("```csv\nid,score\n1,9\n```")

    with sync_patch:
        result = _agent().kickoff("Write it", artifact=True)

    assert result.artifact.kind == "csv"


def test_markdown_with_code_blocks_is_left_alone(store, tmp_path):
    doc = "# Setup\n\n```bash\npip install crewai\n```\n\nThen run it.\n\n```python\nprint(1)\n```"
    sync_patch, _ = _kickoff_patches(doc)

    with sync_patch:
        result = _agent().kickoff("Write it", artifact=True)

    assert result.artifact.kind == "markdown"
    assert (tmp_path / "artifacts" / "exec-1" / "writer" / "v1.md").read_text() == doc


def test_tool_refuses_files_in_the_shared_temp_dir(store):
    import tempfile

    # A sibling temp folder: outside the working directory, inside the system temp.
    other_run = Path(tempfile.mkdtemp()) / "secret.txt"
    other_run.write_text("someone else's data")

    result = SaveArtifactTool().run(
        identifier="leak", title="x", kind="text", file_path=str(other_run)
    )

    assert result.startswith("Error saving artifact")
    assert not (store.root / "exec-1" / "leak").exists()


def test_load_artifact_reads_back_the_latest_or_a_given_version(store):
    from crewai import load_artifact

    report = Artifact(identifier="report", title="Report", kind="html")
    save_artifact(report, content="<p>v1</p>")
    save_artifact(report, content="<p>v2</p>")

    assert load_artifact("report") == b"<p>v2</p>"
    assert load_artifact("report", version=1) == b"<p>v1</p>"


def test_load_artifact_picks_the_highest_version_number(store):
    from crewai import load_artifact

    directory = store.root / "exec-1" / "report"
    directory.mkdir(parents=True)
    (directory / "v9.md").write_text("nine")
    (directory / "v10.md").write_text("ten")

    assert load_artifact("report") == b"ten"


def test_load_artifact_raises_when_missing(store):
    from crewai import load_artifact
    from crewai.artifacts import ArtifactNotFoundError

    save_artifact(Artifact(identifier="report", title="R", kind="html"), content="x")

    with pytest.raises(ArtifactNotFoundError):
        load_artifact("missing")
    with pytest.raises(ArtifactNotFoundError):
        load_artifact("report", version=5)


def test_a_save_only_store_says_it_cannot_load():
    from crewai.artifacts import ArtifactStore

    class SaveOnly(ArtifactStore):
        def save(self, *args, **kwargs):
            raise AssertionError("not called")

    with pytest.raises(NotImplementedError, match="SaveOnly"):
        SaveOnly().load("exec-1", "report")


@pytest.mark.parametrize("identifier", ["../secret", "report/../../x", "Report"])
def test_load_artifact_refuses_identifiers_that_are_not_valid(store, identifier):
    from crewai import load_artifact

    with pytest.raises(ValueError, match="invalid artifact identifier"):
        load_artifact(identifier)


ARTIFACT_INPUT_FLOW = """
schema: crewai.flow/v1
name: ReviseFlow
methods:
  revise:
    start: true
    do:
      call: agent
      with:
        role: Editor
        goal: Revise reports
        backstory: Careful editor.
        input: Revise the report
        input_files:
          current_report: {artifact: board-report}
          first_draft: {artifact: board-report, version: 1}
"""


def test_declarative_agent_reads_artifacts_as_input_files(store):
    report = Artifact(identifier="board-report", title="Board report", kind="html")
    save_artifact(report, content="<p>v1</p>")
    save_artifact(report, content="<p>v2</p>")
    seen = {}

    async def kickoff_async(self, messages, **kwargs):
        seen.update(kwargs["input_files"])
        return "done"

    with patch.object(Agent, "kickoff_async", kickoff_async):
        Flow.from_declaration(contents=ARTIFACT_INPUT_FLOW).kickoff()

    assert seen["current_report"].read() == b"<p>v2</p>"
    assert seen["first_draft"].read() == b"<p>v1</p>"


def test_declarative_artifact_input_that_is_missing_fails_the_step(store):
    async def kickoff_async(self, messages, **kwargs):
        raise AssertionError("must not run without its input")

    with patch.object(Agent, "kickoff_async", kickoff_async), pytest.raises(Exception, match="board-report"):
        Flow.from_declaration(contents=ARTIFACT_INPUT_FLOW).kickoff()


@pytest.mark.parametrize(
    "reference",
    ["{artifact: Bad Id}", "{artifact: board-report, version: 0}", "{artifact: board-report, path: x}", "{file: x}"],
)
def test_declarative_artifact_input_is_validated_when_loaded(reference):
    with pytest.raises(Exception, match="input_files"):
        Flow.from_declaration(contents=ARTIFACT_INPUT_FLOW.replace("{artifact: board-report}", reference))

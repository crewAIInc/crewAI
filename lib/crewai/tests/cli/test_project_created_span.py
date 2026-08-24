"""Scaffolding a project must be observable, and carry the id it just minted.

Acquisition was only observable from a project's first *run*. That misses every project
created and never run, and dates the rest to the wrong day. All three scaffolding paths
already mint a `project_id` into the new `pyproject.toml`; none of them reported it, and
two of them emitted no telemetry at all.

Lives here rather than in `lib/cli/tests/` because `lib/cli/tests/` is not run by any
workflow, while this directory is part of the required job.
"""

from typing import Any
from unittest.mock import patch

from click.testing import CliRunner
from crewai_core.telemetry import Telemetry
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


class _NullExporter:
    """Stands in for the OTLP exporter so no test attempts a real export."""

    def export(self, spans: Any) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture
def recorded_spans(monkeypatch):
    """Spans emitted by a live Telemetry, captured from its own provider.

    Telemetry has to be genuinely enabled: when it is disabled, `__init__` returns before
    building `self.provider` at all, so every assertion here would pass vacuously against
    a non-recording span.

    Enabling it is exactly what makes the exporter swap mandatory. `__init__` wires a
    `BatchSpanProcessor` around the real `SafeOTLPSpanExporter`, pointed at the production
    collector, so without `_NullExporter` these synthetic `Project Created` spans -- with
    invented `created_project_id` values -- are POSTed to it for real. `--block-network`
    does not prevent that: it is function-scoped and only swaps `socket.connect`, which a
    background batch thread outlives. Same reason `_register_shutdown_handlers` is
    suppressed and the provider is shut down in `finally`: otherwise each test leaves an
    atexit hook and an exporter thread behind.

    Follows `telemetry_with_exporter` in tests/telemetry/test_tracer_isolation.py; the
    patch target is `crewai_core` because that is where this Telemetry comes from.
    """
    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)
    for var in ("OTEL_SDK_DISABLED", "CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING"):
        monkeypatch.setenv(var, "false")
    monkeypatch.setattr(
        "crewai_core.telemetry.SafeOTLPSpanExporter",
        lambda **_kwargs: _NullExporter(),
    )

    telemetry = Telemetry()
    assert telemetry.ready, "telemetry must be live or these assertions prove nothing"
    exporter = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        yield telemetry, exporter
    finally:
        telemetry.provider.shutdown()
        Telemetry._instance = None


def test_project_created_span_carries_kind_and_the_minted_id(recorded_spans):
    telemetry, exporter = recorded_spans

    telemetry.project_created_span("crew", "proj-abc123")

    span = next(s for s in exporter.get_finished_spans() if s.name == "Project Created")
    assert span.attributes["kind"] == "crew"
    assert span.attributes["created_project_id"] == "proj-abc123"


def test_the_minted_id_does_not_reuse_the_project_id_attribute(recorded_spans):
    """`project_id` and `created_project_id` mean different things and must stay separate.

    `CommonAttributesSpanProcessor` stamps `project_id` on every span from
    `get_project_id()`, which reads the *current working directory* and is cached for the
    life of the process. During `crewai create` that is the directory the command was run
    from, not the project being created. Writing the minted id into the same attribute
    would silently give one column two meanings.
    """
    telemetry, exporter = recorded_spans

    telemetry.project_created_span("flow", "proj-new")

    span = next(s for s in exporter.get_finished_spans() if s.name == "Project Created")
    assert span.attributes["created_project_id"] == "proj-new"
    assert span.attributes.get("project_id") != "proj-new", (
        "the minted id must not overwrite the cwd-derived project_id"
    )


def test_a_failed_mint_records_an_empty_string_not_a_missing_key(recorded_spans):
    """`get_or_create_project_id` returns None when pyproject is missing or unwritable.

    Empty rather than absent, matching the convention established for the common
    `project_id` attribute: absent means an old client that never sent the key at all,
    which is a different fact from a client that sent it and had nothing to report.
    """
    telemetry, exporter = recorded_spans

    telemetry.project_created_span("crew", None)

    span = next(s for s in exporter.get_finished_spans() if s.name == "Project Created")
    assert span.attributes["created_project_id"] == ""
    assert "created_project_id" in span.attributes


SCAFFOLDERS = [
    ("crew", "crewai_cli.create_crew", "create_crew", {"skip_provider": True}),
    (
        "json_crew",
        "crewai_cli.create_json_crew",
        "create_json_crew",
        {"skip_provider": True},
    ),
    ("flow", "crewai_cli.create_flow", "create_flow", {}),
]


@pytest.mark.parametrize(
    ("kind", "module", "func_name", "kwargs"),
    SCAFFOLDERS,
    ids=[s[0] for s in SCAFFOLDERS],
)
def test_each_scaffolder_reports_the_id_it_minted(
    kind, module, func_name, kwargs, monkeypatch
):
    """The span must carry the value the mint returned, which pins the ordering.

    Asserting on the minted value rather than merely that the span was emitted is what
    makes this a test of ordering: a span emitted before the mint could not carry it.
    `create_flow` already emitted `flow_creation_span` *before* minting, so this is a
    real mistake to guard against and not a hypothetical one.
    """
    import importlib

    # The product's own non-interactive mode. `create_json_crew` otherwise runs an agent
    # wizard that reads stdin; this is the supported way to skip it rather than a stub.
    monkeypatch.setenv("CREWAI_DMN", "1")

    mod = importlib.import_module(module)
    func = getattr(mod, func_name)

    with CliRunner().isolated_filesystem():
        with (
            patch.object(mod, "get_or_create_project_id", return_value="proj-minted"),
            patch.object(mod, "Telemetry") as telemetry_cls,
        ):
            func("demo_project", **kwargs)

    telemetry_cls.return_value.project_created_span.assert_called_once_with(
        kind, "proj-minted"
    )


def test_adding_a_crew_to_an_existing_project_is_not_an_acquisition():
    """`create_crew(parent_folder=...)` adds a crew to a project that already exists.

    It mints no id and must emit no creation span, or every added crew would be counted
    as a newly acquired project.
    """
    from crewai_cli import create_crew as module

    with CliRunner().isolated_filesystem():
        module.create_folder_structure("host_project")
        with (
            patch.object(module, "get_or_create_project_id") as mint,
            patch.object(module, "Telemetry") as telemetry_cls,
        ):
            module.create_crew(
                "extra_crew", skip_provider=True, parent_folder="host_project"
            )

    assert not mint.called, "the parent_folder branch must not mint a new id"
    assert not telemetry_cls.return_value.project_created_span.called, (
        "adding a crew to an existing project must not be reported as a new project"
    )

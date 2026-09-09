"""Checkpoint CLI usage is counted once, without including checkpoint data."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import pytest

from crewai_cli.checkpoint_cli import _record_checkpoint_usage
from crewai_cli.checkpoint_tui import CheckpointTUI
from crewai_cli.cli import crewai
from crewai_core.telemetry import Telemetry


@pytest.mark.parametrize(
    ("args", "target", "action"),
    [
        ([], "checkpoint_tui.run_checkpoint_tui", "tui"),
        (["list", "/private/checkpoints"], "checkpoint_cli.list_checkpoints", "list"),
        (
            ["info", "/private/checkpoint.json"],
            "checkpoint_cli.info_checkpoint",
            "info",
        ),
        (["resume", "private-id"], "checkpoint_cli.resume_checkpoint", "resume"),
        (
            ["diff", "private-id-1", "private-id-2"],
            "checkpoint_cli.diff_checkpoints",
            "diff",
        ),
        (
            ["prune", "--keep", "2", "--dry-run"],
            "checkpoint_cli.prune_checkpoints",
            "prune",
        ),
    ],
)
def test_checkpoint_command_usage(args, target, action):
    with (
        patch("crewai_core.telemetry.Telemetry") as telemetry,
        patch(f"crewai_cli.{target}") as operation,
    ):
        result = CliRunner().invoke(crewai, ["checkpoint", *args])

    assert result.exit_code == 0, result.output
    operation.assert_called_once()
    telemetry.return_value.feature_usage_span.assert_called_once_with(
        f"cli_usage:checkpoint_{action}"
    )


@pytest.mark.parametrize("args", [["--help"], ["list", "--help"], ["diff"]])
def test_help_and_invalid_arguments_are_not_counted(args):
    with patch("crewai_core.telemetry.Telemetry") as telemetry:
        CliRunner().invoke(crewai, ["checkpoint", *args])
    telemetry.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["initialization", "set_tracer", "feature_usage_span"]
)
def test_telemetry_failure_does_not_block_command(failure):
    with (
        patch("crewai_core.telemetry.Telemetry") as telemetry,
        patch("crewai_cli.checkpoint_cli.list_checkpoints") as operation,
    ):
        if failure == "initialization":
            telemetry.side_effect = RuntimeError("telemetry unavailable")
        else:
            getattr(telemetry.return_value, failure).side_effect = RuntimeError(
                "unavailable"
            )
        result = CliRunner().invoke(crewai, ["checkpoint", "list"])

    assert result.exit_code == 0, result.output
    operation.assert_called_once()


@pytest.mark.parametrize(
    "flag", ["OTEL_SDK_DISABLED", "CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING"]
)
def test_checkpoint_usage_respects_telemetry_opt_out(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setattr(Telemetry, "_instance", None)
    telemetry = Telemetry()
    telemetry.provider = MagicMock()
    telemetry.ready = True

    _record_checkpoint_usage("list")

    telemetry.provider.get_tracer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resume", "fork"])
@pytest.mark.parametrize("selected", [False, True])
async def test_tui_counts_only_actions_with_a_selected_checkpoint(action, selected):
    app = CheckpointTUI(location="/nonexistent/checkpoints")
    with (
        patch("crewai_core.telemetry.Telemetry") as telemetry,
        patch.object(app, "_collect_inputs", return_value={}),
        patch.object(app, "_collect_task_overrides", return_value={}),
        patch.object(app, "_resolve_location", return_value="/private/checkpoint.json"),
        patch.object(app, "_detect_entity_type", return_value="crew"),
        patch.object(app, "exit") as exit_app,
    ):
        async with app.run_test():
            app._selected_entry = {"name": "private-id"} if selected else None
            getattr(app, f"action_{action}")()

    if selected:
        telemetry.return_value.feature_usage_span.assert_called_once_with(
            f"cli_usage:checkpoint_tui_{action}"
        )
        exit_app.assert_called_once()
    else:
        telemetry.assert_not_called()
        exit_app.assert_not_called()

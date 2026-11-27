"""checkpoint list/info must honor the group-level --location option."""

from unittest.mock import patch

from click.testing import CliRunner

from crewai_cli.cli import checkpoint


def test_list_uses_group_location():
    with patch("crewai_cli.checkpoint_cli.list_checkpoints") as mocked:
        result = CliRunner().invoke(checkpoint, ["--location", "/tmp/custom", "list"])

    assert result.exit_code == 0, result.output
    assert mocked.call_args.args[0] == "/tmp/custom"


def test_info_uses_group_location():
    with patch("crewai_cli.checkpoint_cli.info_checkpoint") as mocked:
        result = CliRunner().invoke(
            checkpoint, ["--location", "/tmp/custom", "info", "x"]
        )

    assert result.exit_code == 0, result.output
    assert mocked.call_args.args[0] in ("/tmp/custom", "x")

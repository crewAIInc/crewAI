"""Declining an overwrite must report cancellation to the caller."""

from unittest.mock import patch

from crewai_cli.add_crew_to_flow import create_embedded_crew


def test_declined_overwrite_returns_false(tmp_path):
    (tmp_path / "existing").mkdir()

    with patch("crewai_cli.add_crew_to_flow.click.confirm", return_value=False):
        result = create_embedded_crew("existing", tmp_path)

    assert result is False

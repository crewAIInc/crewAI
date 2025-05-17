from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.cli import create


@pytest.fixture
def runner():
    return CliRunner()


class TestCreateCommand:
    @mock.patch("crewai.cli.cli.create_crew")
    def test_create_crew_with_ssl_verify_default(self, mock_create_crew, runner):
        """Test that create crew command passes skip_ssl_verify=False by default."""
        result = runner.invoke(create, ["crew", "test_crew"])

        assert result.exit_code == 0
        mock_create_crew.assert_called_once()
        assert mock_create_crew.call_args[1]["skip_ssl_verify"] is False

    @mock.patch("crewai.cli.cli.create_crew")
    def test_create_crew_with_skip_ssl_verify(self, mock_create_crew, runner):
        """Test that create crew command passes skip_ssl_verify=True when flag is used."""
        result = runner.invoke(create, ["crew", "test_crew", "--skip_ssl_verify"])

        assert result.exit_code == 0
        mock_create_crew.assert_called_once()
        assert mock_create_crew.call_args[1]["skip_ssl_verify"] is True

    @mock.patch("crewai.cli.cli.create_flow")
    def test_create_flow_ignores_skip_ssl_verify(self, mock_create_flow, runner):
        """Test that create flow command ignores the skip_ssl_verify flag."""
        result = runner.invoke(create, ["flow", "test_flow", "--skip_ssl_verify"])

        assert result.exit_code == 0
        mock_create_flow.assert_called_once()
        assert mock_create_flow.call_args == mock.call("test_flow")

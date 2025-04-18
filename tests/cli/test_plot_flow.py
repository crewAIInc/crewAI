import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from crewai.cli.plot_flow import plot_flow


class TestPlotFlow:
    def test_plot_flow_no_main_file(self):
        """Test plot_flow when main.py doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = os.getcwd()
            try:
                os.chdir(temp_dir)
                with patch("click.echo") as mock_echo:
                    plot_flow()
                    mock_echo.assert_called_with(
                        "Error: Could not find main.py in the current directory", err=True
                    )
            finally:
                os.chdir(original_dir)

    def test_plot_flow_with_main_file(self):
        """Test plot_flow with a mock main.py that has plot function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("importlib.util.spec_from_file_location") as mock_spec_from_file:
                with patch("click.echo"):
                    mock_module = MagicMock()
                    mock_spec = MagicMock()
                    mock_spec_from_file.return_value = mock_spec
                    mock_spec.loader = MagicMock()
                    
                    with patch("os.path.exists", return_value=True):
                        plot_flow()
                        mock_spec.loader.exec_module.assert_called_once()

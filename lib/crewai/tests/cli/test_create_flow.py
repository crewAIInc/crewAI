import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.create_flow import RESERVED_FLOW_NAMES, create_flow


@pytest.fixture
def runner():
    return CliRunner()


def test_create_flow_rejects_reserved_flow_names(runner):
    """Test that reserved script names from pyproject.toml are rejected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            for reserved_name in RESERVED_FLOW_NAMES:
                result = runner.invoke(
                    mock.MagicMock(), [], catch_exceptions=False, obj=None
                )

                create_flow(reserved_name)

                folder_path = Path(temp_dir) / reserved_name
                assert not folder_path.exists(), (
                    f"Folder should not be created for reserved name: {reserved_name}"
                )

        finally:
            import os

            os.chdir(original_cwd)


def test_create_flow_rejects_kickoff_name():
    """Test that 'kickoff' name is rejected as it conflicts with pyproject.toml scripts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            create_flow("kickoff")

            folder_path = Path(temp_dir) / "kickoff"
            assert not folder_path.exists(), (
                "Folder should not be created for reserved name: kickoff"
            )

        finally:
            import os

            os.chdir(original_cwd)


def test_create_flow_rejects_plot_name():
    """Test that 'plot' name is rejected as it conflicts with pyproject.toml scripts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            create_flow("plot")

            folder_path = Path(temp_dir) / "plot"
            assert not folder_path.exists(), (
                "Folder should not be created for reserved name: plot"
            )

        finally:
            import os

            os.chdir(original_cwd)


def test_create_flow_allows_valid_names():
    """Test that valid names are allowed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(temp_dir)

            with mock.patch("crewai.cli.create_flow.Telemetry"):
                create_flow("my_valid_flow")

                folder_path = Path(temp_dir) / "my_valid_flow"
                assert folder_path.exists(), "Folder should be created for valid name"

                if folder_path.exists():
                    shutil.rmtree(folder_path)

        finally:
            import os

            os.chdir(original_cwd)

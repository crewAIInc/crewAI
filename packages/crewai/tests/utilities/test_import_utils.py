"""Tests for import utilities."""

import pytest
from unittest.mock import patch

from crewai.utilities.import_utils import require, OptionalDependencyError


class TestRequire:
    """Test the require function."""

    def test_require_existing_module(self):
        """Test requiring a module that exists."""
        module = require("json", purpose="testing")
        assert module.__name__ == "json"

    def test_require_missing_module(self):
        """Test requiring a module that doesn't exist."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require("nonexistent_module_xyz", purpose="testing missing module")

        error_msg = str(exc_info.value)
        assert (
            "testing missing module requires the optional dependency 'nonexistent_module_xyz'"
            in error_msg
        )
        assert "uv add nonexistent_module_xyz" in error_msg

    def test_require_with_import_error(self):
        """Test that ImportError is properly chained."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module import failed")

            with pytest.raises(OptionalDependencyError) as exc_info:
                require("some_module", purpose="testing error handling")

            assert isinstance(exc_info.value.__cause__, ImportError)
            assert str(exc_info.value.__cause__) == "Module import failed"

    def test_optional_dependency_error_is_import_error(self):
        """Test that OptionalDependencyError is a subclass of ImportError."""
        assert issubclass(OptionalDependencyError, ImportError)
